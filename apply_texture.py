import math
import numpy as np
import cv2
import trimesh
from pyproj import Transformer
from PIL import Image
from tqdm import tqdm

# Path to files
IMAGE_PATH = "images/1447902075542541.jpg"
MESH_PATH = "my_region.glb"
OUTPUT_MESH = "my_region_textured.glb"
TEXTURE_SIZE = 512

# Camera parameters
UTM_EPSG = 32617
CAMERA_HEIGHT_M = 1.6
H_FOV_DEG = 65.0
PITCH_DEG = -9.0
ROLL_DEG = 0.0

MAPILLARY = {
    "computed_geometry": {"type": "Point", "coordinates": [-83.743213758351, 42.275425023057]},
    "compass_angle": 179.60961914062,
}

def deg2rad(d):
    return d * math.pi / 180.0

def build_intrinsics(w, h, hfov_deg):
    hfov = deg2rad(hfov_deg)
    fx = (w / 2.0) / math.tan(hfov / 2.0)
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)

def rotation_world_to_camera(yaw_deg, pitch_deg, roll_deg):
    yaw   = deg2rad(yaw_deg)
    pitch = deg2rad(pitch_deg)
    roll  = deg2rad(roll_deg)

    f = np.array([math.sin(yaw), math.cos(yaw), 0.0])
    f /= np.linalg.norm(f)

    up = np.array([0, 0, 1.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)

    R = np.vstack([r, -u, f])

    Rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch),  math.cos(pitch)]])
    Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                   [math.sin(roll),  math.cos(roll), 0],
                   [0, 0, 1]])

    return Rz @ Rx @ R

def camera_center_utm(image_meta):
    lon, lat = image_meta["computed_geometry"]["coordinates"]
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{UTM_EPSG}", always_xy=True)
    x, y = t.transform(lon, lat)
    alt = image_meta.get("computed_altitude", CAMERA_HEIGHT_M)
    return np.array([x, y, alt])

def project(Xw, Cw, R, K):
    Xc = (R @ (Xw - Cw).T).T
    z = Xc[:, 2].copy()
    z[z < 1e-6] = 1e-6
    uv = (K @ np.vstack([Xc[:,0]/z, Xc[:,1]/z, np.ones(len(z))])).T
    return uv[:, :2], z

def convert_mesh_to_utm(mesh_path, origin_lon, origin_lat):
    """Convert mesh from local EPSG:3857 offsets to absolute UTM."""
    print(f"Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path, force="mesh")

    t_merc         = Transformer.from_crs("EPSG:4326", "EPSG:3857",          always_xy=True)
    t_utm          = Transformer.from_crs("EPSG:4326", f"EPSG:{UTM_EPSG}",   always_xy=True)
    t_merc_to_ll   = Transformer.from_crs("EPSG:3857", "EPSG:4326",          always_xy=True)

    origin_merc_x, origin_merc_y = t_merc.transform(origin_lon, origin_lat)

    new_verts = np.zeros_like(mesh.vertices)
    for i, v in enumerate(mesh.vertices):
        lon, lat = t_merc_to_ll.transform(origin_merc_x + v[0], origin_merc_y + v[1])
        utm_x, utm_y = t_utm.transform(lon, lat)
        new_verts[i] = [utm_x, utm_y, v[2]]

    mesh.vertices = new_verts
    return mesh

def apply_photo_texture_to_mesh(mesh, image_path):
    """
    Reproject a photo onto all mesh faces:
      - Visible (camera-facing, in-frame) faces receive photo texture.
      - All other faces receive a flat gray.
    Vertices are unrolled (one unique vertex per face-corner) so every face
    can have its own independent UV without conflicts.
    """

    print(f"\nLoading image: {image_path}")
    real_img = cv2.imread(image_path)
    if real_img is None:
        print("Error: Could not load image")
        return mesh

    h, w = real_img.shape[:2]

    Cw = camera_center_utm(MAPILLARY)
    R  = rotation_world_to_camera(MAPILLARY["compass_angle"], PITCH_DEG, ROLL_DEG)
    K  = build_intrinsics(w, h, H_FOV_DEG)

    vertices     = mesh.vertices
    faces        = mesh.faces
    mesh.face_normals          # ensure computed
    face_normals = mesh.face_normals

    # ------------------------------------------------------------------
    # Find visible faces
    # ------------------------------------------------------------------
    visible_face_indices = []   # indices into `faces`
    face_proj            = []   # (uv_img float32 (3,2), depths (3,))

    print("Finding visible faces with backface culling + occlusion check...")
    for face_idx, face in enumerate(faces):
        v0, v1, v2 = vertices[face]
        tri_verts  = np.array([v0, v1, v2])
        face_center = tri_verts.mean(axis=0)

        view_dir = face_center - Cw
        view_dir /= np.linalg.norm(view_dir)
        if np.dot(face_normals[face_idx], view_dir) >= 0:
            continue  # back-face

        uv_img, depths = project(tri_verts, Cw, R, K)
        if not np.all(depths > 0.1):
            continue

        pts = uv_img.astype(np.int32)
        if not (pts[:, 0].min() >= 0 and pts[:, 0].max() < w and
                pts[:, 1].min() >= 0 and pts[:, 1].max() < h):
            continue

        # Occlusion: cast ray from camera to face center
        ray_len = np.linalg.norm(face_center - Cw)
        ray_dir = (face_center - Cw) / ray_len
        locations, _, index_tri = mesh.ray.intersects_location(
            ray_origins=[Cw], ray_directions=[ray_dir]
        )
        if len(locations) > 0:
            dists        = np.linalg.norm(locations - Cw, axis=1)
            closest_face = index_tri[np.argmin(dists)]
            if closest_face != face_idx and abs(dists.min() - ray_len) >= 0.5:
                continue  # occluded

        visible_face_indices.append(face_idx)
        face_proj.append((uv_img.astype(np.float32), depths))

    print(f"Found {len(visible_face_indices)} visible faces out of {len(faces)} total")

    # ------------------------------------------------------------------
    # Texture atlas layout
    #
    #  Columns  [0 .. photo_col_end]   → reprojected photo (visible faces)
    #  Columns  [gray_col_start .. TS] → flat gray          (all other faces)
    #
    # UV_PHOTO_MAX < 1.0 ensures the rasteriser never touches the gray strip.
    # ------------------------------------------------------------------
    UV_PHOTO_MAX  = 0.97          # visible faces use u ∈ [0, UV_PHOTO_MAX]
    DEFAULT_U     = 0.99          # non-visible faces point here (gray strip)
    DEFAULT_V     = 0.50
    TS            = TEXTURE_SIZE

    gray_col_start = int(UV_PHOTO_MAX * (TS - 1)) + 1

    texture_img = np.full((TS, TS, 3), 255, dtype=np.uint8)
    texture_img[:, gray_col_start:] = [180, 180, 180]   # flat gray strip

    # ------------------------------------------------------------------
    # Unroll ALL faces → one unique vertex per face-corner
    # ------------------------------------------------------------------
    n_all = len(faces)

    new_verts = np.zeros((n_all * 3, 3))
    new_faces = np.arange(n_all * 3, dtype=np.int64).reshape(n_all, 3)
    new_uvs   = np.full((n_all * 3, 2), [DEFAULT_U, DEFAULT_V])   # default = gray

    for i, face in enumerate(faces):
        new_verts[i*3 : i*3+3] = vertices[face]

    # UV normalization bounds (from visible faces only)
    if len(visible_face_indices) == 0:
        print("No visible faces – returning untextured mesh.")
        new_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(texture_img), doubleSided=True)
        new_mesh.visual = trimesh.visual.TextureVisuals(
            uv=new_uvs, material=material, image=Image.fromarray(texture_img))
        return new_mesh

    all_pts   = np.vstack([pts for pts, _ in face_proj])
    img_min_x = all_pts[:, 0].min();  img_max_x = all_pts[:, 0].max()
    img_min_y = all_pts[:, 1].min();  img_max_y = all_pts[:, 1].max()
    img_w     = max(img_max_x - img_min_x, 1e-6)
    img_h     = max(img_max_y - img_min_y, 1e-6)

    print(f"Visible image region: ({img_min_x:.0f},{img_min_y:.0f}) "
          f"→ ({img_max_x:.0f},{img_max_y:.0f})")

    # Assign UV to visible face corners (scale u by UV_PHOTO_MAX)
    for proj_i, face_idx in enumerate(visible_face_indices):
        uv_img, _ = face_proj[proj_i]
        for j in range(3):
            u = (uv_img[j, 0] - img_min_x) / img_w * UV_PHOTO_MAX
            v = (uv_img[j, 1] - img_min_y) / img_h
            new_uvs[face_idx*3 + j] = [u, v]

    # ------------------------------------------------------------------
    # Rasterise photo into texture with perspective-correct sampling
    # ------------------------------------------------------------------
    print(f"Rasterising {len(visible_face_indices)} visible faces into texture "
          f"({TS}×{TS})...")

    for proj_i in tqdm(range(len(visible_face_indices)), desc="Rasterizing texture"):
        face_idx        = visible_face_indices[proj_i]
        uv_img, depths  = face_proj[proj_i]
        uvs             = new_uvs[face_idx*3 : face_idx*3+3]

        # Texture-pixel positions for the 3 corners (v-flipped)
        pts_tex = np.array([
            [uvs[j, 0] * (TS - 1),
             (1.0 - uvs[j, 1]) * (TS - 1)]
            for j in range(3)
        ], dtype=np.float32)

        pts_img_f = uv_img          # (3,2) float32 image coords
        inv_z     = 1.0 / depths    # for perspective-correct interp

        v0t, v1t, v2t = pts_tex
        denom = ((v1t[1]-v2t[1])*(v0t[0]-v2t[0]) +
                 (v2t[0]-v1t[0])*(v0t[1]-v2t[1]))
        if abs(denom) < 1e-6:
            continue

        tx_min = max(0,    int(np.floor(pts_tex[:, 0].min())))
        tx_max = min(TS-1, int(np.ceil (pts_tex[:, 0].max())))
        ty_min = max(0,    int(np.floor(pts_tex[:, 1].min())))
        ty_max = min(TS-1, int(np.ceil (pts_tex[:, 1].max())))

        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                px, py = tx + 0.5, ty + 0.5

                bw0 = ((v1t[1]-v2t[1])*(px-v2t[0]) + (v2t[0]-v1t[0])*(py-v2t[1])) / denom
                bw1 = ((v2t[1]-v0t[1])*(px-v2t[0]) + (v0t[0]-v2t[0])*(py-v2t[1])) / denom
                bw2 = 1.0 - bw0 - bw1

                if bw0 < -0.01 or bw1 < -0.01 or bw2 < -0.01:
                    continue

                # Perspective-correct image-coord interpolation
                interp_inv_z = bw0*inv_z[0] + bw1*inv_z[1] + bw2*inv_z[2]
                img_x = (bw0*pts_img_f[0,0]*inv_z[0] +
                         bw1*pts_img_f[1,0]*inv_z[1] +
                         bw2*pts_img_f[2,0]*inv_z[2]) / interp_inv_z
                img_y = (bw0*pts_img_f[0,1]*inv_z[0] +
                         bw1*pts_img_f[1,1]*inv_z[1] +
                         bw2*pts_img_f[2,1]*inv_z[2]) / interp_inv_z

                ix = int(np.clip(img_x, 0, w - 1))
                iy = int(np.clip(img_y, 0, h - 1))

                color = real_img[iy, ix]
                texture_img[ty, tx] = [color[2], color[1], color[0]]  # BGR→RGB

    print("✓ Texture atlas complete")

    texture_path = 'building_texture_atlas.png'
    Image.fromarray(texture_img).save(texture_path)
    print(f"✓ Saved texture atlas → {texture_path}")

    # ------------------------------------------------------------------
    # Assemble final mesh
    # ------------------------------------------------------------------
    new_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(texture_img),
        doubleSided=True
    )
    new_mesh.visual = trimesh.visual.TextureVisuals(
        uv=new_uvs,
        material=material,
        image=Image.fromarray(texture_img)
    )

    return new_mesh


def main():
    print("="*60)
    print("APPLYING PHOTO TEXTURE TO 3D MESH")
    print("="*60)

    min_lat = min(42.275126, 42.274225)
    max_lat = max(42.275126, 42.274225)
    min_lon = min(-83.744150, -83.743034)
    max_lon = max(-83.744150, -83.743034)

    origin_lon = (min_lon + max_lon) / 2
    origin_lat = (min_lat + max_lat) / 2

    # Convert mesh to UTM for camera-space projection
    mesh_utm = convert_mesh_to_utm(MESH_PATH, origin_lon, origin_lat)

    # Apply texture – returns full unrolled mesh in UTM coordinates
    textured_mesh = apply_photo_texture_to_mesh(mesh_utm, IMAGE_PATH)

    # Re-center: subtract UTM origin so the mesh sits near the world origin
    t_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{UTM_EPSG}", always_xy=True)
    origin_utm_x, origin_utm_y = t_utm.transform(origin_lon, origin_lat)
    textured_mesh.vertices -= np.array([origin_utm_x, origin_utm_y, 0.0])

    bounds = textured_mesh.bounds
    size   = bounds[1] - bounds[0]
    center = textured_mesh.centroid
    print(f"\n  Mesh center : ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) m")
    print(f"  Mesh size   : {size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f} m")

    print(f"\nExporting → {OUTPUT_MESH}")
    textured_mesh.export(OUTPUT_MESH)

    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print(f"  {OUTPUT_MESH}")
    print("  building_texture_atlas.png")

if __name__ == "__main__":
    main()
