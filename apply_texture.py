import math
import numpy as np
import cv2
import trimesh
from pyproj import Transformer
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# Add as many cameras as you like. Each entry needs:
#   image_path        – path to the photo file
#   computed_geometry – GeoJSON Point with [lon, lat]  (use computed_geometry, not geometry)
#
# Rotation — provide EITHER:
#   computed_rotation – Mapillary angle-axis [rx,ry,rz] (preferred, most accurate)
#   OR: compass_angle + pitch_deg + roll_deg  (fallback)
#
# Intrinsics — provide EITHER:
#   camera_parameters – Mapillary [f_norm, k1, k2]  (preferred; f_norm×max(w,h)=fx)
#   OR: hfov_deg                                     (fallback)
#
#   computed_altitude (optional) – camera height in metres; defaults to CAMERA_HEIGHT_M
# ──────────────────────────────────────────────────────────────────────────────
CAMERAS = [
    {
        "image_path": "images/1447902075542541.jpg",
        "computed_geometry": {"type": "Point", "coordinates": [-83.743213758351, 42.275425023057]},
        "compass_angle": 179.60961914062,
        "pitch_deg": -9.0,
        "roll_deg":  0.0,
        "hfov_deg":  65.0,
    },
    {
        "image_path": "images/736076653727528.jpg",
        "computed_geometry": {"type": "Point", "coordinates": [-83.744386553314, 42.274203213453]},
        "computed_rotation": [1.2645213571938, -0.37925136592952, 0.58673461290576],
        "camera_parameters": [0.82404822558687, 0.041268741919332, -0.053786142448042],
        "width": 4032,
        "height": 3024,
    },
    {
        "image_path": "images/2866795086971957.jpg",
        "computed_geometry": {"type": "Point", "coordinates": [-83.744275700379, 42.274181493597]},
        "computed_rotation": [1.1475916354054, -0.77857929538535, 1.0469278214094],
        "camera_parameters": [0.82404822558687, 0.041268741919332, -0.053786142448042],
        "width": 4032,
        "height": 3024,
    },
    # ── paste additional cameras below ────────────────────────────────────────
    # {
    #     "image_path": "images/another_image.jpg",
    #     "computed_geometry": {"type": "Point", "coordinates": [-83.742, 42.275]},
    #     "compass_angle": 90.0,
    #     "pitch_deg": -5.0,
    #     "roll_deg":  0.0,
    #     "hfov_deg":  65.0,
    # },
]

MESH_PATH        = "my_region.glb"
OUTPUT_MESH      = "my_region_textured.glb"
UTM_EPSG         = 32617
CAMERA_HEIGHT_M  = 1.6     # fallback altitude when not in metadata


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
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

def rotation_from_angle_axis(aa):
    """Convert a Mapillary computed_rotation angle-axis vector to a 3×3 matrix.
    Rotates from world (ENU/UTM) coordinates to camera coordinates."""
    aa    = np.array(aa, dtype=np.float64)
    angle = np.linalg.norm(aa)
    if angle < 1e-9:
        return np.eye(3)
    k  = aa / angle
    K  = np.array([[ 0,    -k[2],  k[1]],
                   [ k[2],  0,    -k[0]],
                   [-k[1],  k[0],  0   ]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)

def build_intrinsics_from_params(w, h, camera_parameters):
    """Build K from Mapillary camera_parameters = [f_norm, k1, k2].
    f_norm is normalised by max(w, h); k1/k2 are radial distortion (ignored here)."""
    fx = camera_parameters[0] * max(w, h)
    return np.array([[fx, 0, w/2.0],[0, fx, h/2.0],[0, 0, 1]], dtype=np.float64)

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
        lon, lat = t_m2ll.transform(ox + v[0], oy + v[1])
        ux, uy   = t_ll2u.transform(lon, lat)
        nv[i]    = [ux, uy, v[2]]
    mesh.vertices = nv
    return mesh


# ──────────────────────────────────────────────────────────────────────────────
# Core: multi-camera projection → texture atlas
# ──────────────────────────────────────────────────────────────────────────────
def apply_photo_texture_to_mesh(mesh, cameras):
    """
    For each mesh face, pick the camera that gives the most frontal (least
    oblique) view, then extract the face's texture patch via a true projective
    homography.  All non-visible faces get a neutral gray fallback.

    cameras : list of dicts – see CAMERAS at the top of this file.
    """

    # ── Load images and pre-compute camera matrices ──────────────────────────
    cam_data = []
    for ci, cam in enumerate(cameras):
        img = cv2.imread(cam["image_path"])
        if img is None:
            raise FileNotFoundError(cam["image_path"])
        h, w = img.shape[:2]
        Cw = camera_center_utm(cam)
        if "computed_rotation" in cam:
            R = rotation_from_angle_axis(cam["computed_rotation"])
        else:
            R = rotation_world_to_camera(cam["compass_angle"],
                                         cam.get("pitch_deg", 0.0),
                                         cam.get("roll_deg",  0.0))
        if "camera_parameters" in cam:
            K = build_intrinsics_from_params(w, h, cam["camera_parameters"])
        else:
            K = build_intrinsics(w, h, cam.get("hfov_deg", 65.0))
        cam_data.append({"img": img, "w": w, "h": h, "Cw": Cw, "R": R, "K": K})
        print(f"  Camera {ci}: {cam['image_path']}  ({w}×{h})")

    verts        = mesh.vertices
    faces        = mesh.faces
    _            = mesh.face_normals          # trigger computation
    face_normals = mesh.face_normals

    ATLAS_W   = 4096
    MAX_PATCH = 1024

    # ── Visibility pass: best camera per face ────────────────────────────────
    # best_hit[fi] = (cam_idx, uv_img (3,2), depths (3,), frontality)
    best_hit = {}

    print(f"Visibility pass over {len(cameras)} camera(s) …")
    for ci, cd in enumerate(cam_data):
        Cw, R, K = cd["Cw"], cd["R"], cd["K"]
        w, h     = cd["w"],  cd["h"]

        for fi, face in enumerate(faces):
            V0, V1, V2 = verts[face]
            center = (V0 + V1 + V2) / 3.0
            vd = center - Cw;  vd /= np.linalg.norm(vd)

            # Back-face cull
            frontality = -np.dot(face_normals[fi], vd)   # >0 means facing camera
            if frontality <= 0:
                continue

            uv_img, depths = project(np.array([V0, V1, V2]), Cw, R, K)
            if not np.all(depths > 0.1):
                continue
            pts = uv_img.astype(np.int32)
            if not (pts[:,0].min() >= 0 and pts[:,0].max() < w and
                    pts[:,1].min() >= 0 and pts[:,1].max() < h):
                continue

            # Occlusion test
            ray_len = np.linalg.norm(center - Cw)
            ray_dir = (center - Cw) / ray_len
            locs, _, idx_tri = mesh.ray.intersects_location([Cw], [ray_dir])
            if len(locs) > 0:
                dists = np.linalg.norm(locs - Cw, axis=1)
                if idx_tri[np.argmin(dists)] != fi and abs(dists.min() - ray_len) >= 0.5:
                    continue

            # Keep this camera if it's the most frontal so far for this face
            if fi not in best_hit or frontality > best_hit[fi][3]:
                best_hit[fi] = (ci, uv_img.astype(np.float32), depths, frontality)

    print(f"  {len(best_hit)} visible faces / {len(faces)} total  "
          f"(across {len(cameras)} camera(s))")

    # ── Per-face projective warp → patch ─────────────────────────────────────
    patch_records = []

    print("Extracting per-face patches …")
    for fi, (ci, uv_img, depths, _) in best_hit.items():
        face       = faces[fi]
        V0, V1, V2 = verts[face]
        cd = cam_data[ci]
        Cw, R, K = cd["Cw"], cd["R"], cd["K"]
        real_img = cd["img"]

        # Face-local orthonormal frame
        e1     = V1 - V0
        e1_len = np.linalg.norm(e1)
        if e1_len < 1e-6: continue
        e1_hat = e1 / e1_len

        n     = np.cross(V1 - V0, V2 - V0)
        n_len = np.linalg.norm(n)
        if n_len < 1e-6: continue
        n_hat  = n / n_len
        e2_hat = np.cross(e1_hat, n_hat)
        if e2_hat[2] < 0:
            e2_hat = -e2_hat

        v1_s = np.dot(V1 - V0, e1_hat)
        v1_t = np.dot(V1 - V0, e2_hat)
        v2_s = np.dot(V2 - V0, e1_hat)
        v2_t = np.dot(V2 - V0, e2_hat)

        s_all = [0.0, v1_s, v2_s];  t_all = [0.0, v1_t, v2_t]
        s_min, s_max = min(s_all), max(s_all)
        t_min, t_max = min(t_all), max(t_all)
        if s_max <= s_min or t_max <= t_min: continue

        # Consistent px/m so neither axis is stretched
        face_w_m = s_max - s_min
        face_h_m = t_max - t_min
        img_bw   = max(uv_img[:,0].max() - uv_img[:,0].min(), 1.0)
        img_bh   = max(uv_img[:,1].max() - uv_img[:,1].min(), 1.0)
        px_per_m = min(img_bw / face_w_m, img_bh / face_h_m)
        patch_w  = int(np.clip(face_w_m * px_per_m, 2, MAX_PATCH))
        patch_h  = int(np.clip(face_h_m * px_per_m, 2, MAX_PATCH))

        # Projective homography: face-local (s,t) → image pixel
        M = np.column_stack([R @ e1_hat, R @ e2_hat, R @ (V0 - Cw)])
        H = K @ M

        # Atlas pixel (ax,ay) → face-local (s,t), ay=0 = top of face
        ds = (s_max - s_min) / patch_w
        dt = (t_max - t_min) / patch_h
        S  = np.array([[ds,  0,   s_min],
                       [ 0, -dt,  t_max],
                       [ 0,  0,   1    ]], dtype=np.float64)
        H_atlas = H @ S

        patch_bgr = cv2.warpPerspective(
            real_img, H_atlas, (patch_w, patch_h),
            flags      = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
            borderMode = cv2.BORDER_REPLICATE)
        patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)

        def to_px(s, t, _ds=ds, _dt=dt, _s_min=s_min, _t_max=t_max):
            return (s - _s_min) / _ds, (_t_max - t) / _dt

        patch_records.append({
            'fi':    fi,
            'patch': patch_rgb,
            'pw':    patch_w,
            'ph':    patch_h,
            'v0_px': to_px(0.0,  0.0 ),
            'v1_px': to_px(v1_s, v1_t),
            'v2_px': to_px(v2_s, v2_t),
        })

    # ── Pack patches into atlas ───────────────────────────────────────────────
    cx, cy, row_h = 0, 0, 0
    for p in patch_records:
        if cx + p['pw'] > ATLAS_W:
            cx = 0; cy += row_h; row_h = 0
        p['ax'] = cx;  p['ay'] = cy
        cx += p['pw'];  row_h = max(row_h, p['ph'])
    photo_h = cy + row_h

    GRAY_H  = 16
    atlas_h = photo_h + GRAY_H
    atlas   = np.full((atlas_h, ATLAS_W, 3), 180, dtype=np.uint8)
    for p in patch_records:
        atlas[p['ay']:p['ay']+p['ph'], p['ax']:p['ax']+p['pw']] = p['patch']

    DEFAULT_U = 0.5
    DEFAULT_V = 1.0 - (photo_h + GRAY_H / 2) / atlas_h

    # ── Unroll mesh, assign UVs ───────────────────────────────────────────────
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

    # ── Export ────────────────────────────────────────────────────────────────
    tex_pil = Image.fromarray(atlas)
    tex_pil.save("building_texture_atlas.png")
    print(f"✓ Atlas saved  ({ATLAS_W}×{atlas_h} px, {len(patch_records)} face patches)")

    new_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=tex_pil, doubleSided=True)
    new_mesh.visual = trimesh.visual.TextureVisuals(uv=new_uvs, material=mat, image=tex_pil)
    return new_mesh


# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    min_lat = min(42.275126, 42.274225);  max_lat = max(42.275126, 42.274225)
    min_lon = min(-83.744150,-83.743034); max_lon = max(-83.744150,-83.743034)
    origin_lon = (min_lon + max_lon) / 2
    origin_lat = (min_lat + max_lat) / 2

    mesh_utm      = convert_mesh_to_utm(MESH_PATH, origin_lon, origin_lat)
    textured_mesh = apply_photo_texture_to_mesh(mesh_utm, CAMERAS)

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
