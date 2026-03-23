import math
import numpy as np
import cv2
import trimesh
from pyproj import Transformer
from PIL import Image

# Path to files
IMAGE_PATH  = "images/1447902075542541.jpg"
MESH_PATH   = "my_region.glb"
OUTPUT_MESH = "my_region_textured.glb"

# Camera parameters
UTM_EPSG       = 32617
CAMERA_HEIGHT_M = 1.6
H_FOV_DEG      = 65.0
PITCH_DEG      = -9.0
ROLL_DEG       = 0.0

MAPILLARY = {
    "computed_geometry": {"type": "Point", "coordinates": [-83.743213758351, 42.275425023057]},
    "compass_angle": 179.60961914062,
}

def deg2rad(d):
    return d * math.pi / 180.0

def build_intrinsics(w, h, hfov_deg):
    hfov = deg2rad(hfov_deg)
    fx   = (w / 2.0) / math.tan(hfov / 2.0)
    cx, cy = w / 2.0, h / 2.0
    return np.array([[fx, 0, cx],
                     [0, fx, cy],
                     [0,  0,  1]], dtype=np.float64)

def rotation_world_to_camera(yaw_deg, pitch_deg, roll_deg):
    yaw   = deg2rad(yaw_deg)
    pitch = deg2rad(pitch_deg)
    roll  = deg2rad(roll_deg)

    f = np.array([math.sin(yaw), math.cos(yaw), 0.0])
    f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    r  = np.cross(f, up);  r /= np.linalg.norm(r)
    u  = np.cross(r, f)

    R  = np.vstack([r, -u, f])
    Rx = np.array([[1, 0, 0],
                   [0,  math.cos(pitch), -math.sin(pitch)],
                   [0,  math.sin(pitch),  math.cos(pitch)]])
    Rz = np.array([[ math.cos(roll), -math.sin(roll), 0],
                   [ math.sin(roll),  math.cos(roll), 0],
                   [0, 0, 1]])
    return Rz @ Rx @ R

def camera_center_utm(meta):
    lon, lat = meta["computed_geometry"]["coordinates"]
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{UTM_EPSG}", always_xy=True)
    x, y = t.transform(lon, lat)
    return np.array([x, y, meta.get("computed_altitude", CAMERA_HEIGHT_M)])

def project(Xw, Cw, R, K):
    Xc = (R @ (Xw - Cw).T).T
    z  = Xc[:, 2].copy();  z[z < 1e-6] = 1e-6
    uv = (K @ np.vstack([Xc[:,0]/z, Xc[:,1]/z, np.ones(len(z))])).T
    return uv[:, :2], z

def convert_mesh_to_utm(mesh_path, origin_lon, origin_lat):
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh")

    t_merc       = Transformer.from_crs("EPSG:4326", "EPSG:3857",        always_xy=True)
    t_utm        = Transformer.from_crs("EPSG:4326", f"EPSG:{UTM_EPSG}", always_xy=True)
    t_merc_to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326",        always_xy=True)

    ox, oy = t_merc.transform(origin_lon, origin_lat)
    new_v  = np.zeros_like(mesh.vertices)
    for i, v in enumerate(mesh.vertices):
        lon, lat   = t_merc_to_ll.transform(ox + v[0], oy + v[1])
        ux, uy     = t_utm.transform(lon, lat)
        new_v[i]   = [ux, uy, v[2]]
    mesh.vertices = new_v
    return mesh


def apply_photo_texture_to_mesh(mesh, image_path):
    """
    Reproject a photo onto the mesh at native resolution.

    Strategy
    --------
    * Project every visible face's vertices to image pixel coordinates.
    * Crop that exact region from the original image (no resize, no atlas
      rasterisation).
    * Append a narrow gray strip to the right of the crop to serve as the
      fallback colour for non-visible faces.
    * UVs are computed directly from the projection → normalised into the
      crop texture.  The GPU's own perspective-correct interpolation handles
      the rest.
    * All faces are kept (mesh unrolled so every face has independent UVs).
    """

    print(f"\nLoading image: {image_path}")
    real_img = cv2.imread(image_path)
    if real_img is None:
        raise FileNotFoundError(f"Cannot load {image_path}")
    img_h, img_w = real_img.shape[:2]

    Cw = camera_center_utm(MAPILLARY)
    R  = rotation_world_to_camera(MAPILLARY["compass_angle"], PITCH_DEG, ROLL_DEG)
    K  = build_intrinsics(img_w, img_h, H_FOV_DEG)

    vertices     = mesh.vertices
    faces        = mesh.faces
    mesh.face_normals               # ensure computed
    face_normals = mesh.face_normals

    # ------------------------------------------------------------------
    # 1. Find visible faces + project their vertices
    # ------------------------------------------------------------------
    visible_idx  = []               # indices into `faces`
    face_proj    = []               # (uv_img float32 (3,2), depths (3,))

    print("Finding visible faces (backface cull + occlusion)...")
    for fi, face in enumerate(faces):
        v0, v1, v2  = vertices[face]
        tri         = np.array([v0, v1, v2])
        center      = tri.mean(axis=0)

        vd = center - Cw;  vd /= np.linalg.norm(vd)
        if np.dot(face_normals[fi], vd) >= 0:
            continue                             # back-face

        uv_img, depths = project(tri, Cw, R, K)
        if not np.all(depths > 0.1):
            continue

        pts = uv_img.astype(np.int32)
        if not (pts[:,0].min() >= 0 and pts[:,0].max() < img_w and
                pts[:,1].min() >= 0 and pts[:,1].max() < img_h):
            continue

        # Occlusion: ray to face centre
        ray_len = np.linalg.norm(center - Cw)
        ray_dir = (center - Cw) / ray_len
        locs, _, idx_tri = mesh.ray.intersects_location(
            ray_origins=[Cw], ray_directions=[ray_dir])
        if len(locs) > 0:
            dists = np.linalg.norm(locs - Cw, axis=1)
            hit   = idx_tri[np.argmin(dists)]
            if hit != fi and abs(dists.min() - ray_len) >= 0.5:
                continue                         # occluded

        visible_idx.append(fi)
        face_proj.append((uv_img.astype(np.float32), depths))

    print(f"  {len(visible_idx)} visible / {len(faces)} total faces")

    # ------------------------------------------------------------------
    # 2. Build texture: native-res crop  +  gray fallback strip
    # ------------------------------------------------------------------
    if len(visible_idx) == 0:
        print("No visible faces – skipping texture.")
        # Return untextured mesh at original coords
        return mesh

    all_pts  = np.vstack([p for p, _ in face_proj])
    cx0 = max(0,       int(np.floor(all_pts[:,0].min())))
    cy0 = max(0,       int(np.floor(all_pts[:,1].min())))
    cx1 = min(img_w-1, int(np.ceil (all_pts[:,0].max())))
    cy1 = min(img_h-1, int(np.ceil (all_pts[:,1].max())))

    crop_w = cx1 - cx0
    crop_h = cy1 - cy0
    print(f"  Photo crop: ({cx0},{cy0})→({cx1},{cy1})  size {crop_w}×{crop_h} px")

    # Crop at native resolution (BGR→RGB)
    photo_crop = cv2.cvtColor(real_img[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2RGB)

    # Gray strip (8 px wide) appended on the right
    GRAY_W     = 8
    gray_strip = np.full((crop_h, GRAY_W, 3), 180, dtype=np.uint8)
    texture    = np.hstack([photo_crop, gray_strip])   # (crop_h, crop_w+GRAY_W, 3)

    tex_w = texture.shape[1]   # = crop_w + GRAY_W
    tex_h = texture.shape[0]   # = crop_h

    # UV of the gray strip centre (for non-visible faces)
    DEFAULT_U = (crop_w + GRAY_W / 2) / tex_w
    DEFAULT_V = 0.5

    print(f"  Texture size: {tex_w}×{tex_h} px  (photo {crop_w} + gray {GRAY_W})")

    # ------------------------------------------------------------------
    # 3. Unroll ALL faces; assign UVs directly from the projection
    # ------------------------------------------------------------------
    n_all    = len(faces)
    new_verts = np.zeros((n_all * 3, 3))
    new_faces = np.arange(n_all * 3, dtype=np.int64).reshape(n_all, 3)
    new_uvs   = np.full((n_all * 3, 2), [DEFAULT_U, DEFAULT_V])

    # Copy geometry
    for i, face in enumerate(faces):
        new_verts[i*3 : i*3+3] = vertices[face]

    # Assign UVs for visible faces directly from projection
    # u = (proj_x - cx0) / tex_w          (relative to crop left)
    # v = 1 - (proj_y - cy0) / tex_h      (v-flip: image y=0 is top)
    for proj_i, fi in enumerate(visible_idx):
        uv_img, _ = face_proj[proj_i]
        for j in range(3):
            u = (uv_img[j, 0] - cx0) / tex_w
            v = 1.0 - (uv_img[j, 1] - cy0) / tex_h
            new_uvs[fi*3 + j] = [np.clip(u, 0.0, crop_w / tex_w),
                                  np.clip(v, 0.0, 1.0)]

    # ------------------------------------------------------------------
    # 4. Build and return the textured mesh
    # ------------------------------------------------------------------
    texture_pil = Image.fromarray(texture)
    texture_pil.save("building_texture_atlas.png")
    print("✓ Saved native-res crop texture → building_texture_atlas.png")

    new_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    material  = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture_pil, doubleSided=True)
    new_mesh.visual = trimesh.visual.TextureVisuals(
        uv=new_uvs, material=material, image=texture_pil)

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

    mesh_utm      = convert_mesh_to_utm(MESH_PATH, origin_lon, origin_lat)
    textured_mesh = apply_photo_texture_to_mesh(mesh_utm, IMAGE_PATH)

    # Re-centre: subtract UTM origin so mesh sits near the world origin
    t_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{UTM_EPSG}", always_xy=True)
    ox, oy = t_utm.transform(origin_lon, origin_lat)
    textured_mesh.vertices -= np.array([ox, oy, 0.0])

    b = textured_mesh.bounds
    print(f"\n  Size: {b[1]-b[0]}")
    print(f"  Centre: {textured_mesh.centroid}")

    print(f"\nExporting → {OUTPUT_MESH}")
    textured_mesh.export(OUTPUT_MESH)

    print("\n" + "="*60)
    print(f"Done.  {OUTPUT_MESH}  |  building_texture_atlas.png")
    print("="*60)

if __name__ == "__main__":
    main()
