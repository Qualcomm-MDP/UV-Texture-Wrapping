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
TEXTURE_SIZE = 512  # Reduced from 2048 for 4x faster processing

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
    yaw = deg2rad(yaw_deg)
    pitch = deg2rad(pitch_deg)
    roll = deg2rad(roll_deg)

    f = np.array([math.sin(yaw), math.cos(yaw), 0.0])
    f /= np.linalg.norm(f)

    up = np.array([0, 0, 1.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)

    R = np.vstack([r, -u, f])

    Rx = np.array([[1,0,0],
                   [0,math.cos(pitch),-math.sin(pitch)],
                   [0,math.sin(pitch), math.cos(pitch)]])
    Rz = np.array([[math.cos(roll),-math.sin(roll),0],
                   [math.sin(roll), math.cos(roll),0],
                   [0,0,1]])

    return Rz @ Rx @ R

def camera_center_utm(image_meta):
    lon, lat = image_meta["computed_geometry"]["coordinates"]
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{UTM_EPSG}", always_xy=True)
    x, y = t.transform(lon, lat)
    alt = image_meta.get("computed_altitude", CAMERA_HEIGHT_M)
    return np.array([x, y, alt])

def project(Xw, Cw, R, K):
    Xc = (R @ (Xw - Cw).T).T
    z = Xc[:, 2]
    z[z < 1e-6] = 1e-6
    uv = (K @ np.vstack([Xc[:,0]/z, Xc[:,1]/z, np.ones(len(z))])).T
    return uv[:, :2], z

def convert_mesh_to_utm(mesh_path, origin_lon, origin_lat):
    """Convert mesh from EPSG:3857 to UTM"""
    print(f"Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path, force="mesh")
    
    t_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    t_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{UTM_EPSG}", always_xy=True)
    
    origin_merc_x, origin_merc_y = t_merc.transform(origin_lon, origin_lat)
    
    vertices = mesh.vertices.copy()
    new_vertices = []
    
    for v in vertices:
        abs_merc_x = origin_merc_x + v[0]
        abs_merc_y = origin_merc_y + v[1]
        
        t_merc_to_latlon = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        lon, lat = t_merc_to_latlon.transform(abs_merc_x, abs_merc_y)
        
        utm_x, utm_y = t_utm.transform(lon, lat)
        new_vertices.append([utm_x, utm_y, v[2]])
    
    mesh.vertices = np.array(new_vertices)
    return mesh

def apply_photo_texture_to_mesh(mesh, image_path, crop_path='building_texture_crop.png'):
    """Apply cropped photo texture to visible mesh faces"""
    
    print(f"\nLoading image: {image_path}")
    real_img = cv2.imread(image_path)
    if real_img is None:
        print(f"Error: Could not load image")
        return mesh
    
    print(f"Loading crop: {crop_path}")
    crop_img = cv2.imread(crop_path, cv2.IMREAD_UNCHANGED)
    if crop_img is None:
        print(f"Error: Could not load crop")
        return mesh
    
    crop_h, crop_w = crop_img.shape[:2]
    h, w = real_img.shape[:2]
    
    # Convert crop to RGB
    if crop_img.shape[2] == 4:
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGRA2RGB)
        crop_mask = crop_img[:, :, 3]
    else:
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        crop_mask = None
    
    # Camera setup
    Cw = camera_center_utm(MAPILLARY)
    R = rotation_world_to_camera(MAPILLARY["compass_angle"], PITCH_DEG, ROLL_DEG)
    K = build_intrinsics(w, h, H_FOV_DEG)
    
    print(f"Creating texture atlas ({TEXTURE_SIZE}x{TEXTURE_SIZE})...")
    
    # Create texture image with white background
    texture_img = np.full((TEXTURE_SIZE, TEXTURE_SIZE, 3), 255, dtype=np.uint8)
    
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Calculate face normals
    print(f"Calculating face normals and visibility...")
    mesh.face_normals  # This ensures normals are computed
    face_normals = mesh.face_normals
    
    # Find visible faces and their image coordinates
    visible_faces = []
    face_img_coords = []
    
    print(f"Finding visible faces with backface culling and occlusion check...")
    for face_idx, face in enumerate(faces):
        v0, v1, v2 = vertices[face]
        tri_verts = np.array([v0, v1, v2])
        
        # Calculate face center
        face_center = tri_verts.mean(axis=0)
        
        # Check if face is facing the camera (backface culling)
        view_direction = face_center - Cw
        view_direction = view_direction / np.linalg.norm(view_direction)
        
        face_normal = face_normals[face_idx]
        dot_product = np.dot(face_normal, view_direction)
        
        # If dot product < 0, face normal points toward camera (face is visible)
        if dot_product >= 0:
            continue  # Face is facing away from camera, skip it
        
        # Project to image
        uv_img, depths = project(tri_verts, Cw, R, K)
        
        # Check if visible and in bounds
        if np.all(depths > 0.1):
            pts_img = uv_img.astype(np.int32)
            
            if (pts_img[:, 0].min() >= 0 and pts_img[:, 0].max() < w and
                pts_img[:, 1].min() >= 0 and pts_img[:, 1].max() < h):
                
                # Additional occlusion check using ray casting
                ray_direction = face_center - Cw
                ray_length = np.linalg.norm(ray_direction)
                ray_direction = ray_direction / ray_length
                
                # Cast ray from camera to face center
                locations, index_ray, index_tri = mesh.ray.intersects_location(
                    ray_origins=[Cw],
                    ray_directions=[ray_direction]
                )
                
                # Check if this face is the first hit (not occluded)
                if len(locations) > 0:
                    # Find the closest intersection
                    distances = np.linalg.norm(locations - Cw, axis=1)
                    closest_idx = np.argmin(distances)
                    closest_face = index_tri[closest_idx]
                    
                    # Only add if this face is the closest hit (within tolerance)
                    if closest_face == face_idx or abs(distances[closest_idx] - ray_length) < 0.5:
                        visible_faces.append(face_idx)
                        face_img_coords.append(pts_img)
                else:
                    # No intersection found (shouldn't happen), add it anyway
                    visible_faces.append(face_idx)
                    face_img_coords.append(pts_img)
    
    print(f"Found {len(visible_faces)} visible faces")
    
    if len(visible_faces) == 0:
        print("No visible faces found!")
        return mesh
    
    # Calculate bounding box of all visible faces in image space
    all_coords = np.vstack(face_img_coords)
    img_min_x = all_coords[:, 0].min()
    img_max_x = all_coords[:, 0].max()
    img_min_y = all_coords[:, 1].min()
    img_max_y = all_coords[:, 1].max()
    
    img_width = img_max_x - img_min_x
    img_height = img_max_y - img_min_y
    
    print(f"Visible region in image: ({img_min_x}, {img_min_y}) to ({img_max_x}, {img_max_y})")
    print(f"Size: {img_width}x{img_height}")
    
    # Create new UV coordinates
    new_uvs = np.zeros((len(vertices), 2))
    
    # For each visible face, map its image coordinates to UV space
    for i, face_idx in enumerate(visible_faces):
        face = faces[face_idx]
        pts_img = face_img_coords[i]
        
        # Normalize image coordinates to [0, 1] based on visible region
        uv_coords = []
        for pt in pts_img:
            u = (pt[0] - img_min_x) / img_width if img_width > 0 else 0.5
            v = (pt[1] - img_min_y) / img_height if img_height > 0 else 0.5
            uv_coords.append([u, v])
        
        # Assign UV coordinates to vertices
        for j, vertex_idx in enumerate(face):
            new_uvs[vertex_idx] = uv_coords[j]
    
    # Now create texture by sampling from the original image
    print(f"Sampling texture from image...")
    
    for face_idx in tqdm(visible_faces, desc="Rasterizing texture"):
        face = faces[face_idx]
        
        # Get UV coordinates for this face
        uv0, uv1, uv2 = new_uvs[face]
        
        # Convert UV to texture pixel coordinates
        pts_tex = np.array([
            [int(uv0[0] * (TEXTURE_SIZE - 1)), int((1 - uv0[1]) * (TEXTURE_SIZE - 1))],
            [int(uv1[0] * (TEXTURE_SIZE - 1)), int((1 - uv1[1]) * (TEXTURE_SIZE - 1))],
            [int(uv2[0] * (TEXTURE_SIZE - 1)), int((1 - uv2[1]) * (TEXTURE_SIZE - 1))]
        ], dtype=np.float32)
        
        # Get original image coordinates
        idx = visible_faces.index(face_idx)
        pts_img = face_img_coords[idx].astype(np.float32)
        
        # Create transformation matrix to warp from texture to image
        # We'll fill the texture triangle by sampling from the image triangle
        
        # Get bounding box of triangle in texture space
        tex_min_x = max(0, int(pts_tex[:, 0].min()))
        tex_max_x = min(TEXTURE_SIZE - 1, int(pts_tex[:, 0].max()))
        tex_min_y = max(0, int(pts_tex[:, 1].min()))
        tex_max_y = min(TEXTURE_SIZE - 1, int(pts_tex[:, 1].max()))
        
        # Rasterize triangle
        for ty in range(tex_min_y, tex_max_y + 1):
            for tx in range(tex_min_x, tex_max_x + 1):
                # Check if pixel is inside triangle using barycentric coordinates
                p = np.array([tx, ty], dtype=np.float32)
                v0 = pts_tex[0]
                v1 = pts_tex[1]
                v2 = pts_tex[2]
                
                denom = ((v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1]))
                if abs(denom) < 1e-6:
                    continue
                
                w0 = ((v1[1] - v2[1]) * (p[0] - v2[0]) + (v2[0] - v1[0]) * (p[1] - v2[1])) / denom
                w1 = ((v2[1] - v0[1]) * (p[0] - v2[0]) + (v0[0] - v2[0]) * (p[1] - v2[1])) / denom
                w2 = 1 - w0 - w1
                
                if w0 >= -0.01 and w1 >= -0.01 and w2 >= -0.01:
                    # Interpolate image coordinates using barycentric weights
                    img_x = w0 * pts_img[0, 0] + w1 * pts_img[1, 0] + w2 * pts_img[2, 0]
                    img_y = w0 * pts_img[0, 1] + w1 * pts_img[1, 1] + w2 * pts_img[2, 1]
                    
                    # Sample from image
                    ix = int(np.clip(img_x, 0, w - 1))
                    iy = int(np.clip(img_y, 0, h - 1))
                    
                    color = real_img[iy, ix]
                    texture_img[ty, tx] = [color[2], color[1], color[0]]  # BGR to RGB
    
    print(f"✓ Created texture atlas from photo")
    
    # Save texture image
    texture_path = 'building_texture_atlas.png'
    Image.fromarray(texture_img).save(texture_path)
    print(f"✓ Saved texture atlas to {texture_path}")
    
    # Create material with texture
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(texture_img),
        doubleSided=True
    )
    
    # Apply texture to mesh
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=new_uvs,
        material=material,
        image=Image.fromarray(texture_img)
    )
    
    return mesh

def main():
    print("="*60)
    print("APPLYING PHOTO TEXTURE TO 3D MESH")
    print("="*60)
    
    # Mesh parameters from main.py
    min_lat = min(42.275126, 42.274225)
    max_lat = max(42.275126, 42.274225)
    min_lon = min(-83.744150, -83.743034)
    max_lon = max(-83.744150, -83.743034)
    
    origin_lon = (min_lon + max_lon) / 2
    origin_lat = (min_lat + max_lat) / 2
    
    # Load original mesh (in local EPSG:3857 coordinates)
    print(f"Loading original mesh: {MESH_PATH}")
    mesh_original = trimesh.load(MESH_PATH, force="mesh")
    
    # Convert mesh to UTM for texture projection
    mesh_utm = convert_mesh_to_utm(MESH_PATH, origin_lon, origin_lat)
    
    # Apply texture from photo (uses UTM coordinates for camera projection)
    textured_mesh = apply_photo_texture_to_mesh(mesh_utm, IMAGE_PATH)
    
    # Transfer texture back to original mesh (centered at origin)
    print("\nConverting back to local coordinates (centered at origin)...")
    mesh_original.visual = textured_mesh.visual  # Copy texture/UVs
    
    # Calculate bounds for info
    bounds = mesh_original.bounds
    center = mesh_original.centroid
    size = bounds[1] - bounds[0]
    
    print(f"  Mesh center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) meters")
    print(f"  Mesh size: {size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f} meters")
    print(f"  X range: [{bounds[0][0]:.1f}, {bounds[1][0]:.1f}]")
    print(f"  Y range: [{bounds[0][1]:.1f}, {bounds[1][1]:.1f}]")
    print(f"  Z range: [{bounds[0][2]:.1f}, {bounds[1][2]:.1f}]")
    
    # Export textured mesh in local coordinates
    print(f"\nExporting textured mesh to {OUTPUT_MESH}...")
    mesh_original.export(OUTPUT_MESH)
    
    print("\n" + "="*60)
    print("TEXTURE APPLICATION COMPLETE!")
    print("="*60)
    print(f"Output: {OUTPUT_MESH}")
    print("       building_texture_atlas.png")
    print("\nMesh is in local coordinates centered near origin.")
    print("You can now view the textured 3D model in Blender or any GLB viewer!")

if __name__ == "__main__":
    main()
