#!/usr/bin/env python3
# Generating male  die pairs (with clearance and mirrored handles)
# The female die is generated in a separate file.
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
    # set longer side to grid_px
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
    # for relief: darker -> higher
    inv = (1.0 - arr) * mask_arr
    return inv, target_w, target_h, aspect

def build_die_mesh(heightfield, target_w, target_h, aspect, output_stl,
                   longest_side_mm=25.0, relief_mm=2.4, plate_thickness_mm=5.0,
                   handle_diam_mm=16.0, handle_height_mm=20.0, handle_segs=20,
                   clearance_mm=0.1, inverse=False, handle_side='back', grid_pad=2):
    """
    heightfield: 2D array 0..1 where 1 -> max relief
    inverse: if False -> male (positive relief on top), if True -> female (negative cavity)
    handle_side: 'back' means handle attached to bottom of plate (below), 'front' attached above plate
    """
    # physical sizes
    target_w = int(target_w); target_h = int(target_h)
    img_w_mm = longest_side_mm
    img_h_mm = longest_side_mm * (target_h / target_w)
    pad = grid_pad
    full_w = target_w + 2*pad
    full_h = target_h + 2*pad
    border_margin_ratio = 0.0  # no frame
    plate_w = img_w_mm * (1 + 2*border_margin_ratio)
    plate_h = img_h_mm * (1 + 2*border_margin_ratio)
    xs_full = np.linspace(-plate_w/2, plate_w/2, full_w)
    ys_full = np.linspace(-plate_h/2, plate_h/2, full_h)
    xxf, yyf = np.meshgrid(xs_full, ys_full)
    # prepare z grid
    z_full = np.zeros((full_h, full_w), dtype=np.float32)
    # place relief in center
    # male: positive relief on top z>0 up to relief_mm
    # female: cavity: negative depths down to -relief_mm - clearance
    if not inverse:
        # male
        z_center = heightfield * relief_mm  # 0..relief_mm
        z_full[pad:pad+target_h, pad:pad+target_w] = z_center
    else:
        # female: cavity deeper by clearance; we model top surface at 0 and cavity goes negative
        z_center = - (heightfield * relief_mm + clearance_mm)  # negative values
        z_full[pad:pad+target_h, pad:pad+target_w] = z_center
    # Build top surface vertices
    verts = []
    idx_map = -np.ones((full_h, full_w), dtype=int)
    for i in range(full_h):
        for j in range(full_w):
            verts.append([float(xxf[i,j]), float(yyf[i,j]), float(z_full[i,j])])
            idx_map[i,j] = len(verts)-1
    # bottom plate vertices at z = -plate_thickness_mm (for male and female both)
    bottom_start = len(verts)
    for i in range(full_h):
        for j in range(full_w):
            verts.append([float(xxf[i,j]), float(yyf[i,j]), float(-plate_thickness_mm)])
    faces = []
    # top triangulation
    for i in range(full_h-1):
        for j in range(full_w-1):
            v00 = int(idx_map[i,j]); v10 = int(idx_map[i,j+1]); v01 = int(idx_map[i+1,j]); v11 = int(idx_map[i+1,j+1])
            faces.append([v00, v10, v11]); faces.append([v00, v11, v01])
    def bottom_idx(i,j): return bottom_start + i*full_w + j
    # side walls (connect top perimeter to bottom)
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
        faces.append([top_a, bot_a, bot_b]); faces.append([top_a, bot_b, top_b])
    # bottom face triangulation
    for i in range(full_h-1):
        for j in range(full_w-1):
            v00 = bottom_idx(i,j); v10 = bottom_idx(i,j+1); v01 = bottom_idx(i+1,j); v11 = bottom_idx(i+1,j+1)
            faces.append([v00, v11, v10]); faces.append([v00, v01, v11])
    # add handle: for mirrored orientation, if handle_side=='back' attach to bottom (z = -plate_thickness_mm downward),
    # if 'front' attach to top (z = plate top + something)
    handle_radius = handle_diam_mm/2.0
    # determine handle top z depending on side
    if handle_side == 'back':
        handle_top_z = -plate_thickness_mm  # attach at bottom surface
        cyl_height = handle_height_mm  # extend downward
        cyl_bottom_z = handle_top_z - cyl_height
    else:  # 'front'
        # need top of plate: find max z in top surface to place handle above it slightly
        top_surface_z = np.max(z_full)
        handle_top_z = top_surface_z + 0.0  # attach at top surface level
        cyl_height = handle_height_mm
        cyl_bottom_z = handle_top_z + cyl_height  # extend upward
    nseg = handle_segs
    theta = np.linspace(0, 2*math.pi, nseg, endpoint=False)
    vstart = len(verts)
    # create top and bottom rings for cylinder
    for t in theta:
        x = handle_radius * math.cos(t); y = handle_radius * math.sin(t)
        verts.append([x,y,handle_top_z])
    for t in theta:
        x = handle_radius * math.cos(t); y = handle_radius * math.sin(t)
        verts.append([x,y,cyl_bottom_z])
    # cap point for rounding
    cap_idx = len(verts)
    if handle_side == 'back':
        verts.append([0.0,0.0,cyl_bottom_z - handle_radius*0.2])
    else:
        verts.append([0.0,0.0,cyl_bottom_z + handle_radius*0.2])
    # cylinder faces
    top_start = vstart; bot_start = vstart + nseg
    for k in range(nseg):
        a = top_start + k; b = top_start + ((k+1)%nseg); c = bot_start + k; d = bot_start + ((k+1)%nseg)
        faces.append([a,c,d]); faces.append([a,d,b])
    # connect top ring to nearest plate face by fan (simple approach)
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
    # connect bottom ring to cap
    for k in range(nseg):
        a = bot_start + k; b = bot_start + ((k+1)%nseg)
        faces.append([a, cap_idx, b])
    # write STL
    verts = np.array(verts, dtype=np.float32)
    def write_ascii_stl(filename, verts, faces, name="die"):
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

# Paths and parameters
ann_img = "/mnt/data/GaLKxtxWQAA09iq.jpeg"

# make heightfields
hf1, w1, h1, asp1 = make_heightfield_from_image(ann_img, crop_bottom_ratio=0.30, remove_white_bg=False, remove_strings=False, grid_px=300)

# generate male (positive) with handle on back, female (negative) with handle on front (mirrored)
out_files = []
s1 = build_die_mesh(hf1, w1, h1, asp1, "/mnt/data/google_anniversary_male_die.stl",
                    longest_side_mm=25.0, relief_mm=2.4, plate_thickness_mm=5.0,
                    handle_diam_mm=16.0, handle_height_mm=20.0, handle_segs=24,
                    clearance_mm=0.1, inverse=False, handle_side='back', grid_pad=2)
out_files.append(("/mnt/data/google_anniversary_male_die.stl", s1))

print("Created files:")
for p,s in out_files:
    print(p, round(s/1024.0,1), "KB")
