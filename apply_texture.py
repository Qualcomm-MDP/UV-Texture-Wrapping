import math
import numpy as np
import cv2
import trimesh
from pyproj import Transformer
from PIL import Image

IMAGE_PATH  = "images/1447902075542541.jpg"
MESH_PATH   = "my_region.glb"
OUTPUT_MESH = "my_region_textured.glb"

UTM_EPSG        = 32617
CAMERA_HEIGHT_M = 1.6
H_FOV_DEG       = 65.0
PITCH_DEG       = -9.0
ROLL_DEG        = 0.0

MAPILLARY = {
    "computed_geometry": {"type": "Point", "coordinates": [-83.743213758351, 42.275425023057]},
    "compass_angle": 179.60961914062,
}

def deg2rad(d): return d * math.pi / 180.0

def build_intrinsics(w, h, hfov_deg):
    fx = (w / 2.0) / math.tan(deg2rad(hfov_deg) / 2.0)
    return np.array([[fx, 0, w/2.0],[0, fx, h/2.0],[0, 0, 1]], dtype=np.float64)

def rotation_world_to_camera(yaw_deg, pitch_deg, roll_deg):
    yaw, pitch, roll = deg2rad(yaw_deg), deg2rad(pitch_deg), deg2rad(roll_deg)
    f = np.array([math.sin(yaw), math.cos(yaw), 0.0]); f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R  = np.vstack([r, -u, f])
    Rx = np.array([[1,0,0],[0,math.cos(pitch),-math.sin(pitch)],[0,math.sin(pitch),math.cos(pitch)]])
    Rz = np.array([[math.cos(roll),-math.sin(roll),0],[math.sin(roll),math.cos(roll),0],[0,0,1]])
    return Rz @ Rx @ R

def camera_center_utm(meta):
    lon, lat = meta["computed_geometry"]["coordinates"]
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{UTM_EPSG}", always_xy=True)
    x, y = t.transform(lon, lat)
    return np.array([x, y, meta.get("computed_altitude", CAMERA_HEIGHT_M)])

def project(Xw, Cw, R, K):
    Xc = (R @ (Xw - Cw).T).T
    z  = Xc[:, 2].copy(); z[z < 1e-6] = 1e-6
    uv = (K @ np.vstack([Xc[:,0]/z, Xc[:,1]/z, np.ones(len(z))])).T
    return uv[:, :2], z

def convert_mesh_to_utm(mesh_path, origin_lon, origin_lat):
    mesh = trimesh.load(mesh_path, force="mesh")
    t_m2ll = Transformer.from_crs("EPSG:3857", "EPSG:4326",        always_xy=True)
    t_ll2u = Transformer.from_crs("EPSG:4326", f"EPSG:{UTM_EPSG}", always_xy=True)
    t_ll2m = Transformer.from_crs("EPSG:4326", "EPSG:3857",        always_xy=True)
    ox, oy = t_ll2m.transform(origin_lon, origin_lat)
    nv = np.zeros_like(mesh.vertices)
    for i, v in enumerate(mesh.vertices):
        lon, lat   = t_m2ll.transform(ox + v[0], oy + v[1])
        ux, uy     = t_ll2u.transform(lon, lat)
        nv[i]      = [ux, uy, v[2]]
    mesh.vertices = nv
    return mesh


def apply_photo_texture_to_mesh(mesh, image_path):
    """
    Per-face projective texture extraction.

    For each visible face:
      1. Build the true 3-D→image projective homography H = K @ [R·e1, R·e2, R·(V0-Cw)]
      2. Compose with a pixel-scale matrix to get  H_atlas: atlas-pixel → image-pixel
      3. cv2.warpPerspective extracts the face's image patch without distortion
      4. Pack patches into an atlas; assign UVs from the patch corners
    All other faces share a gray fallback region.
    """
    print(f"Loading image: {image_path}")
    real_img = cv2.imread(image_path)
    if real_img is None:
        raise FileNotFoundError(image_path)
    img_h, img_w = real_img.shape[:2]

    Cw = camera_center_utm(MAPILLARY)
    R  = rotation_world_to_camera(MAPILLARY["compass_angle"], PITCH_DEG, ROLL_DEG)
    K  = build_intrinsics(img_w, img_h, H_FOV_DEG)

    verts        = mesh.vertices
    faces        = mesh.faces
    mesh.face_normals
    face_normals = mesh.face_normals

    # ------------------------------------------------------------------
    # 1. Visibility pass
    # ------------------------------------------------------------------
    visible_idx = []
    face_proj   = []          # (uv_img (3,2) float32, depths (3,) float64)

    print("Visibility pass …")
    for fi, face in enumerate(faces):
        V0, V1, V2 = verts[face]
        center = (V0 + V1 + V2) / 3.0
        vd = center - Cw;  vd /= np.linalg.norm(vd)
        if np.dot(face_normals[fi], vd) >= 0:
            continue

        uv_img, depths = project(np.array([V0, V1, V2]), Cw, R, K)
        if not np.all(depths > 0.1):
            continue
        pts = uv_img.astype(np.int32)
        if not (pts[:,0].min() >= 0 and pts[:,0].max() < img_w and
                pts[:,1].min() >= 0 and pts[:,1].max() < img_h):
            continue

        ray_len = np.linalg.norm(center - Cw)
        ray_dir = (center - Cw) / ray_len
        locs, _, idx_tri = mesh.ray.intersects_location([Cw], [ray_dir])
        if len(locs) > 0:
            dists = np.linalg.norm(locs - Cw, axis=1)
            if idx_tri[np.argmin(dists)] != fi and abs(dists.min() - ray_len) >= 0.5:
                continue

        visible_idx.append(fi)
        face_proj.append((uv_img.astype(np.float32), depths))

    print(f"  {len(visible_idx)} visible / {len(faces)} total")
    if not visible_idx:
        return mesh

    # ------------------------------------------------------------------
    # 2. Per-face projective warp → patch
    # ------------------------------------------------------------------
    ATLAS_W  = 4096
    MAX_PATCH = 1024   # cap per-face patch dimensions

    patch_records = []   # dicts filled below

    print("Extracting per-face patches …")
    for proj_i, fi in enumerate(visible_idx):
        face       = faces[fi]
        V0, V1, V2 = verts[face]

        # Face-local orthonormal frame: e1 along V0→V1, e2 perp in face plane
        e1      = V1 - V0
        e1_len  = np.linalg.norm(e1)
        if e1_len < 1e-6: continue
        e1_hat  = e1 / e1_len

        n       = np.cross(V1 - V0, V2 - V0)
        n_len   = np.linalg.norm(n)
        if n_len < 1e-6: continue
        n_hat   = n / n_len
        # e2 points "up" along the face (for a wall, roughly +Z)
        e2_hat  = np.cross(e1_hat, n_hat)   # right-hand: down the face
        # flip so e2 points upward along the building face
        if e2_hat[2] < 0:
            e2_hat = -e2_hat

        # Local (s, t) of each vertex   (metres, e2 increasing upward)
        v1_s  = np.dot(V1 - V0, e1_hat)   # = e1_len
        v1_t  = np.dot(V1 - V0, e2_hat)   # ≈ 0 for horizontal edge
        v2_s  = np.dot(V2 - V0, e1_hat)
        v2_t  = np.dot(V2 - V0, e2_hat)

        s_all = [0.0, v1_s, v2_s];  t_all = [0.0, v1_t, v2_t]
        s_min, s_max = min(s_all), max(s_all)
        t_min, t_max = min(t_all), max(t_all)
        if s_max <= s_min or t_max <= t_min: continue

        # Patch pixel size: match the image-space bounding box of the projection
        pts_img  = face_proj[proj_i][0]
        patch_w  = int(np.clip(pts_img[:,0].max() - pts_img[:,0].min(), 2, MAX_PATCH))
        patch_h  = int(np.clip(pts_img[:,1].max() - pts_img[:,1].min(), 2, MAX_PATCH))

        # True projective homography: face local (s,t) → image pixel (homogeneous)
        #   H @ [s, t, 1]^T  =  K @ R @ (V0 + s·e1_hat + t·e2_hat − Cw)
        M = np.column_stack([R @ e1_hat, R @ e2_hat, R @ (V0 - Cw)])
        H = K @ M                          # 3×3, face-local → image

        # S_offset: atlas pixel (ax, ay) → face local (s, t)
        # s = s_min + ax · ds,   t = t_max − ay · dt  (ay=0 = top = largest t)
        ds = (s_max - s_min) / patch_w
        dt = (t_max - t_min) / patch_h
        S  = np.array([[ds, 0,  s_min ],
                       [ 0, -dt, t_max],   # flip t so ay=0 is top of face
                       [ 0,  0,  1    ]], dtype=np.float64)

        H_atlas = H @ S   # atlas pixel → image (homogeneous)

        # Extract patch from real image
        patch_bgr = cv2.warpPerspective(
            real_img, H_atlas, (patch_w, patch_h),
            flags      = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
            borderMode = cv2.BORDER_REPLICATE)
        patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)

        # Vertex atlas-pixel positions (matching the S above)
        def to_px(s, t):
            ax = (s - s_min) / ds
            ay = (t_max - t) / dt    # flipped t
            return (ax, ay)

        patch_records.append({
            'fi':      fi,
            'patch':   patch_rgb,
            'pw':      patch_w,
            'ph':      patch_h,
            'v0_px':   to_px(0.0,  0.0 ),
            'v1_px':   to_px(v1_s, v1_t),
            'v2_px':   to_px(v2_s, v2_t),
        })

    # ------------------------------------------------------------------
    # 3. Pack patches into atlas (simple row packing)
    # ------------------------------------------------------------------
    cx, cy, row_h = 0, 0, 0
    for p in patch_records:
        if cx + p['pw'] > ATLAS_W:
            cx = 0; cy += row_h; row_h = 0
        p['ax'] = cx;  p['ay'] = cy
        cx += p['pw'];  row_h = max(row_h, p['ph'])
    photo_h = cy + row_h

    # Gray fallback strip below the photo rows
    GRAY_H   = 16
    atlas_h  = photo_h + GRAY_H
    atlas    = np.full((atlas_h, ATLAS_W, 3), 180, dtype=np.uint8)
    for p in patch_records:
        atlas[p['ay']:p['ay']+p['ph'], p['ax']:p['ax']+p['pw']] = p['patch']

    DEFAULT_U = 0.5
    DEFAULT_V = 1.0 - (photo_h + GRAY_H / 2) / atlas_h   # centre of gray strip

    # ------------------------------------------------------------------
    # 4. Build unrolled mesh with per-face UVs
    # ------------------------------------------------------------------
    n_all     = len(faces)
    new_verts = np.zeros((n_all * 3, 3))
    new_faces = np.arange(n_all * 3, dtype=np.int64).reshape(n_all, 3)
    new_uvs   = np.full((n_all * 3, 2), [DEFAULT_U, DEFAULT_V])

    for i, face in enumerate(faces):
        new_verts[i*3:i*3+3] = verts[face]

    for p in patch_records:
        fi = p['fi']
        for j, key in enumerate(('v0_px', 'v1_px', 'v2_px')):
            ax_v, ay_v = p[key]
            u = (p['ax'] + ax_v) / ATLAS_W
            v = 1.0 - (p['ay'] + ay_v) / atlas_h
            new_uvs[fi*3 + j] = [np.clip(u, 0, 1), np.clip(v, 0, 1)]

    # ------------------------------------------------------------------
    # 5. Export
    # ------------------------------------------------------------------
    tex_pil = Image.fromarray(atlas)
    tex_pil.save("building_texture_atlas.png")
    print(f"✓ Atlas saved  ({ATLAS_W}×{atlas_h} px, {len(patch_records)} face patches)")

    new_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=tex_pil, doubleSided=True)
    new_mesh.visual = trimesh.visual.TextureVisuals(uv=new_uvs, material=mat, image=tex_pil)
    return new_mesh


def main():
    print("="*60)
    min_lat = min(42.275126, 42.274225);  max_lat = max(42.275126, 42.274225)
    min_lon = min(-83.744150,-83.743034); max_lon = max(-83.744150,-83.743034)
    origin_lon = (min_lon + max_lon) / 2
    origin_lat = (min_lat + max_lat) / 2

    mesh_utm      = convert_mesh_to_utm(MESH_PATH, origin_lon, origin_lat)
    textured_mesh = apply_photo_texture_to_mesh(mesh_utm, IMAGE_PATH)

    t_utm  = Transformer.from_crs("EPSG:4326", f"EPSG:{UTM_EPSG}", always_xy=True)
    ox, oy = t_utm.transform(origin_lon, origin_lat)
    textured_mesh.vertices -= np.array([ox, oy, 0.0])
    center = textured_mesh.bounding_box.centroid
    textured_mesh.apply_translation(-center)
    rotation = trimesh.transformations.rotation_matrix(
        angle=np.radians(-90.0),
        direction=[1.0, 0.0, 0.0],
        point=[0.0, 0.0, 0.0],
    )

    textured_mesh.apply_transform(rotation)
    print(f"Exporting → {OUTPUT_MESH}")
    textured_mesh.export(OUTPUT_MESH)
    print("Done.")

if __name__ == "__main__":
    main()
