#!/usr/bin/env python3
# Regenerating only the two female dies with the handle attached to the back (opposite the artwork).
from PIL import Image, ImageOps, ImageFilter
import numpy as np, math, os, traceback

def make_heightfield_from_image(image_path, crop_bottom_ratio=0.0, remove_white_bg=False, remove_strings=False, grid_px=300):
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size
    if crop_bottom_ratio > 0:
        img = img.crop((0,0,w,int(h*(1-crop_bottom_ratio))))
    if remove_white_bg:
        arr = np.asarray(img).astype(np.uint8)
        r,g,b,a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]
        white_thresh = 245
        bg_mask = (r > white_thresh) & (g > white_thresh) & (b > white_thresh)
        arr[:,:,3] = np.where(bg_mask, 0, a)
        img = Image.fromarray(arr, mode="RGBA")
    alpha = np.asarray(img)[:,:,3].astype(np.uint8)
    content_mask = (alpha > 10).astype(np.uint8) * 255
    mask_img = Image.fromarray(content_mask, mode="L")
    if remove_strings:
        mask_img = mask_img.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
        mask_img = mask_img.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    rgb = img.convert("RGB")
    aspect = rgb.width / rgb.height
    if aspect >= 1:
        target_w = grid_px
        target_h = int(round(grid_px / aspect))
    else:
        target_h = grid_px
        target_w = int(round(grid_px * aspect))
    gray = ImageOps.grayscale(rgb).resize((target_w, target_h), Image.LANCZOS)
    mask_resized = mask_img.resize((target_w, target_h), Image.LANCZOS)
    arr = np.asarray(gray).astype(np.float32) / 255.0
    mask_arr = np.asarray(mask_resized).astype(np.float32) / 255.0
    inv = (1.0 - arr) * mask_arr
    return inv, target_w, target_h, aspect

def build_die_mesh(heightfield, target_w, target_h, aspect, output_stl,
                   longest_side_mm=25.0, relief_mm=2.4, plate_thickness_mm=5.0,
                   handle_diam_mm=16.0, handle_height_mm=20.0, handle_segs=20,
                   clearance_mm=0.1, inverse=False, handle_side='back', grid_pad=2):
    target_w = int(target_w); target_h = int(target_h)
    img_w_mm = longest_side_mm
    img_h_mm = longest_side_mm * (target_h / target_w)
    pad = grid_pad
    full_w = target_w + 2*pad
    full_h = target_h + 2*pad
    plate_w = img_w_mm
    plate_h = img_h_mm
    xs_full = np.linspace(-plate_w/2, plate_w/2, full_w)
    ys_full = np.linspace(-plate_h/2, plate_h/2, full_h)
    xxf, yyf = np.meshgrid(xs_full, ys_full)
    z_full = np.zeros((full_h, full_w), dtype=np.float32)
    if not inverse:
        z_center = heightfield * relief_mm
        z_full[pad:pad+target_h, pad:pad+target_w] = z_center
    else:
        z_center = - (heightfield * relief_mm + clearance_mm)
        z_full[pad:pad+target_h, pad:pad+target_w] = z_center
    verts = []
    idx_map = -np.ones((full_h, full_w), dtype=int)
    for i in range(full_h):
        for j in range(full_w):
            verts.append([float(xxf[i,j]), float(yyf[i,j]), float(z_full[i,j])])
            idx_map[i,j] = len(verts)-1
    bottom_start = len(verts)
    for i in range(full_h):
        for j in range(full_w):
            verts.append([float(xxf[i,j]), float(yyf[i,j]), float(-plate_thickness_mm)])
    faces = []
    for i in range(full_h-1):
        for j in range(full_w-1):
            v00 = int(idx_map[i,j]); v10 = int(idx_map[i,j+1]); v01 = int(idx_map[i+1,j]); v11 = int(idx_map[i+1,j+1])
            faces.append([v00, v10, v11]); faces.append([v00, v11, v01])
    def bottom_idx(i,j): return bottom_start + i*full_w + j
    for j in range(full_w-1):
        top_a = int(idx_map[0,j]); top_b = int(idx_map[0,j+1]); bot_a = bottom_idx(0,j); bot_b = bottom_idx(0,j+1)
        faces.append([top_a, bot_a, bot_b]); faces.append([top_a, bot_b, top_b])
    for i in range(full_h-1):
        top_a = int(idx_map[i, full_w-1]); top_b = int(idx_map[i+1, full_w-1]); bot_a = bottom_idx(i, full_w-1); bot_b = bottom_idx(i+1, full_w-1)
        faces.append([top_a, bot_a, bot_b]); faces.append([top_a, bot_b, top_b])
    for j in range(full_w-1, 0, -1):
        top_a = int(idx_map[full_h-1, j]); top_b = int(idx_map[full_h-1, j-1]); bot_a = bottom_idx(full_h-1, j); bot_b = bottom_idx(full_h-1, j-1)
        faces.append([top_a, bot_a, bot_b]); faces.append([top_a, bot_b, top_b])
    for i in range(full_h-1, 0, -1):
        top_a = int(idx_map[i,0]); top_b = int(idx_map[i-1,0]); bot_a = bottom_idx(i,0); bot_b = bottom_idx(i-1,0)
        faces.append([top_a, bot_a, bot_b]); faces.append([top_a, bot_b, top_top if False else top_b])
    for i in range(full_h-1):
        for j in range(full_w-1):
            v00 = bottom_idx(i,j); v10 = bottom_idx(i,j+1); v01 = bottom_idx(i+1,j); v11 = bottom_idx(i+1,j+1)
            faces.append([v00, v11, v10]); faces.append([v00, v01, v11])
    # add simple solid cylindrical handle attached to BACK of plate so it's opposite the artwork face
    handle_radius = handle_diam_mm/2.0
    handle_top_z = -plate_thickness_mm  # attach at bottom surface
    cyl_height = handle_height_mm
    cyl_bottom_z = handle_top_z - cyl_height
    nseg = handle_segs
    theta = np.linspace(0, 2*math.pi, nseg, endpoint=False)
    vstart = len(verts)
    for t in theta:
        x = handle_radius * math.cos(t); y = handle_radius * math.sin(t)
        verts.append([x,y,handle_top_z])
    for t in theta:
        x = handle_radius * math.cos(t); y = handle_radius * math.sin(t)
        verts.append([x,y,cyl_bottom_z])
    verts.append([0.0,0.0,cyl_bottom_z - handle_radius*0.3])
    cap_idx = len(verts)-1
    top_start = vstart; bot_start = vstart + nseg
    for k in range(nseg):
        a = top_start + k; b = top_start + ((k+1)%nseg); c = bot_start + k; d = bot_start + ((k+1)%nseg)
        faces.append([a,c,d]); faces.append([a,d,b])
    xs_full_arr = xs_full; ys_full_arr = ys_full
    def find_nearest_plate_index(x,y):
        ix = int(round((x - xs_full_arr[0]) / (xs_full_arr[-1]-xs_full_arr[0]) * (full_w-1)))
        iy = int(round((y - ys_full_arr[0]) / (ys_full_arr[-1]-ys_full_arr[0]) * (full_h-1)))
        ix = min(max(ix,0), full_w-1); iy = min(max(iy,0), full_h-1)
        return bottom_start + iy*full_w + ix
    for k in range(nseg):
        a = top_start + k
        nearest = find_nearest_plate_index(verts[a][0], verts[a][1])
        faces.append([nearest, a, top_start + ((k+1)%nseg)])
    for k in range(nseg):
        a = bot_start + k; b = bot_start + ((k+1)%nseg)
        faces.append([a, cap_idx, b])
    verts = np.array(verts, dtype=np.float32)
    # write stl
    def write_ascii_stl(filename, verts, faces, name="die_fixed"):
        with open(filename, "w") as f:
            f.write(f"solid {name}\n")
            for tri in faces:
                v0 = verts[tri[0]]; v1 = verts[tri[1]]; v2 = verts[tri[2]]
                u = v1 - v0; v = v2 - v0
                n = np.cross(u, v)
                norm = np.linalg.norm(n)
                if norm == 0:
                    nx, ny, nz = 0.0, 0.0, 0.0
                else:
                    nx, ny, nz = n / norm
                f.write(f"  facet normal {nx:.6f} {ny:.6f} {nz:.6f}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
                f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
                f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write(f"endsolid {name}\n")
    write_ascii_stl(output_stl, verts, faces, name=os.path.splitext(os.path.basename(output_stl))[0])
    return os.path.getsize(output_stl)

# Regenerate female dies with handle attached to BACK of plate so art side is unobstructed.
ann_img = "/mnt/data/GaLKxtxWQAA09iq.jpeg"

hf1, w1, h1, asp1 = make_heightfield_from_image(ann_img, crop_bottom_ratio=0.30, remove_white_bg=False, remove_strings=False, grid_px=300)

out1 = "/mnt/data/google_anniversary_female_die_fixed.stl"

s1 = build_die_mesh(hf1, w1, h1, asp1, out1, longest_side_mm=25.0, relief_mm=2.4, plate_thickness_mm=5.0,
                    handle_diam_mm=16.0, handle_height_mm=20.0, handle_segs=24, clearance_mm=0.1,
                    inverse=True, handle_side='back', grid_pad=2)

