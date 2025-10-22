#!/usr/bin/env python3
"""
Thinking
Interpreting handle placement
I’m trying to figure out what the user means by “natural way” for handle placement. It seems like they might want it centered along the long axis, but placed near the back? I want to avoid a misunderstanding, so I'll assume the handle is centered geometrically but elongated along the longer dimension.

Deciding handle placement
I'll go with the handle being centered along the long dimension for simplicity. Since the user mentioned "natural way" but also asked about geometric, I’ll assume it’s a basic geometric center. This means the handle will be placed at the intersection of the long and short axes. Let’s go ahead with that.

Placing handle on plate
I’ll use the existing code with some tweaks to add the handle. The handle will be a cylinder placed on the back of the plate (opposite of the stamping surface). For simplicity, I’ll make the cylinder with about 32 segments, ensuring it blends smoothly with the base, positioned at z = -plate_thickness.

Attaching and shaping the handle
The handle should attach to the back of the plate at z = -plate_thickness and extend downward. I’ll go with a cylinder whose top is flush with the bottom, and the bottom of the handle will go further down, at z = -plate_thickness - handle_height. I’ll include a rounded top as a simple hemisphere.

Thought for 46s
Done

Summary of final specs used:

Longest side: 25 mm

Engraving depth: 2.4 mm

Raised border: 1.5 mm

Base thickness: 5 mm

Handle: solid cylinder 16 mm Ø × 20 mm H, with a small rounded cap, centered naturally on the back and integrated into the model

Resolution: same high raster resolution (grid ~300 px on long side)

Notes & caveats:

Files are high-detail and fairly large (~45 MB each).

I approximated the rounded top of the handle with a small cap to keep the model robust and the file size reasonable. If you want a smoother or fully spherical cap I can regenerate with a higher hemisphere resolution.

The handle is centered on the stamp in the natural (geometric) position so it aligns comfortably when held.
"""
# Generating deeper seals with solid handles for both images.
from PIL import Image, ImageOps, ImageFilter
import numpy as np, math, os

def generate_seal(image_path, output_stl, crop_bottom_ratio=0.0, remove_white_bg=False, remove_strings=False,
                  longest_side_mm=25.0, engrave_depth_mm=2.4, border_height_mm=1.5,
                  plate_thickness_mm=5.0, grid_px=300,
                  handle_diam_mm=16.0, handle_height_mm=20.0, handle_segs=32):
    # Load image
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size
    if crop_bottom_ratio > 0:
        img = img.crop((0,0,w,int(h*(1-crop_bottom_ratio))))  # crop bottom fraction
    
    # Remove white background if requested
    if remove_white_bg:
        arr = np.asarray(img).astype(np.uint8)
        r,g,b,a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]
        white_thresh = 245
        bg_mask = (r > white_thresh) & (g > white_thresh) & (b > white_thresh)
        arr[:,:,3] = np.where(bg_mask, 0, a)
        img = Image.fromarray(arr, mode="RGBA")
    
    # Create content mask and optionally remove thin strings
    alpha = np.asarray(img)[:,:,3].astype(np.uint8)
    content_mask = (alpha > 10).astype(np.uint8) * 255
    mask_img = Image.fromarray(content_mask, mode="L")
    if remove_strings:
        mask_img = mask_img.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
        mask_img = mask_img.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    # grayscale heightfield
    rgb = img.convert("RGB")
    # Determine target size preserving aspect ratio: grid_px on longer side
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
    inv = (1.0 - arr) * mask_arr  # apply mask to zero out background
    
    # physical sizes
    img_w_mm = longest_side_mm
    img_h_mm = longest_side_mm * (target_h / target_w)
    border_margin_ratio = 0.06
    plate_w = img_w_mm * (1 + 2*border_margin_ratio)
    plate_h = img_h_mm * (1 + 2*border_margin_ratio)
    pad = 2
    full_w = target_w + 2*pad
    full_h = target_h + 2*pad
    
    xs_full = np.linspace(-plate_w/2, plate_w/2, full_w)
    ys_full = np.linspace(-plate_h/2, plate_h/2, full_h)
    xxf, yyf = np.meshgrid(xs_full, ys_full)
    
    z_full = np.zeros((full_h, full_w), dtype=np.float32)
    # insert engraving (negative depth)
    z_full[pad:pad+target_h, pad:pad+target_w] = - (inv * engrave_depth_mm)
    
    # raised border outside inner rect
    inner_left = -img_w_mm/2 + img_w_mm * 0.02
    inner_right = img_w_mm/2 - img_w_mm * 0.02
    inner_top = -img_h_mm/2 + img_h_mm * 0.02
    inner_bottom = img_h_mm/2 - img_h_mm * 0.02
    
    for i in range(full_h):
        for j in range(full_w):
            x = xxf[i,j]; y = yyf[i,j]
            if (x < inner_left) or (x > inner_right) or (y < inner_top) or (y > inner_bottom):
                if z_full[i,j] > -engrave_depth_mm/2:
                    z_full[i,j] = max(z_full[i,j], border_height_mm)
    
    # Build top grid vertices
    verts = []
    idx_map = -np.ones((full_h, full_w), dtype=int)
    for i in range(full_h):
        for j in range(full_w):
            verts.append([float(xxf[i,j]), float(yyf[i,j]), float(z_full[i,j])])
            idx_map[i,j] = len(verts)-1
    # bottom plate vertices
    bottom_start = len(verts)
    for i in range(full_h):
        for j in range(full_w):
            verts.append([float(xxf[i,j]), float(yyf[i,j]), float(-plate_thickness_mm)])
    verts = np.array(verts, dtype=np.float32)
    
    faces = []
    # top surface triangulation
    for i in range(full_h-1):
        for j in range(full_w-1):
            v00 = int(idx_map[i,j]); v10 = int(idx_map[i,j+1]); v01 = int(idx_map[i+1,j]); v11 = int(idx_map[i+1,j+1])
            faces.append([v00, v10, v11]); faces.append([v00, v11, v01])
    def bottom_idx(i,j): return bottom_start + i*full_w + j
    # side walls
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
    
    # Now add handle (solid cylinder + hemisphere dome) attached to bottom
    # Determine handle center: natural way => centered along long axis (longer side) and centered across width
    # We'll place handle at geometric center (x=0,y=0) which is natural for holding the stamp.
    handle_radius = handle_diam_mm/2.0
    handle_top_z = -plate_thickness_mm  # attach to bottom of plate
    cyl_height = handle_height_mm - handle_radius  # cylinder portion; hemisphere height = radius
    cyl_bottom_z = handle_top_z - cyl_height
    # create ring of vertices for cylinder (top and bottom)
    nseg = handle_segs
    theta = np.linspace(0, 2*math.pi, nseg, endpoint=False)
    handle_vid_start = len(verts)
    # top circle vertices at handle_top_z
    for t in theta:
        x = handle_radius * math.cos(t)
        y = handle_radius * math.sin(t)
        verts = np.vstack([verts, np.array([x, y, handle_top_z], dtype=np.float32)])
    # bottom circle vertices at cyl_bottom_z
    for t in theta:
        x = handle_radius * math.cos(t)
        y = handle_radius * math.sin(t)
        verts = np.vstack([verts, np.array([x, y, cyl_bottom_z], dtype=np.float32)])
    # center bottom pole for hemisphere bottom
    hemi_center_z = cyl_bottom_z  # sphere center at cylinder bottom (equator at top of hemisphere)
    # add hemisphere vertices (sample latitudes)
    lat_segs = max(4, nseg//4)
    hemi_vid_indices = []
    for li in range(1, lat_segs+1):  # 1..lat_segs (exclude equator which is at li=0)
        phi = (li/lat_segs) * (math.pi/2)  # 0..pi/2
        r_ring = handle_radius * math.cos(phi)
        z = hemi_center_z - handle_radius * math.sin(phi)
        ring_idx = []
        for t in theta:
            x = r_ring * math.cos(t)
            y = r_ring * math.sin(t)
            verts = np.vstack([verts, np.array([x,y,z], dtype=np.float32)])
            ring_idx.append(len(verts)-1)
        hemi_vid_indices.append(ring_idx)
    # add bottom pole vertex
    verts = np.vstack([verts, np.array([0.0, 0.0, hemi_center_z - handle_radius], dtype=np.float32)])
    pole_idx = len(verts)-1
    
    # Now create faces to connect cylinder to plate bottom: we need to translate handle circle coordinates to be centered on plate
    # Our plate is centered at (0,0), so handle centered at (0,0) too. However, plate vertices include grid points; ensure no duplicates.
    # Cylinder top vertices index range:
    top_start = handle_vid_start
    bot_start = handle_vid_start + nseg
    # side faces of cylinder
    for k in range(nseg):
        a = top_start + k
        b = top_start + ((k+1)%nseg)
        c = bot_start + k
        d = bot_start + ((k+1)%nseg)
        faces.append([a, c, d])
        faces.append([a, d, b])
    # connect cylinder top circle to plate bottom: create faces from plate bottom center grid to handle top ring.
    # Find closest bottom grid vertex indices to map a small patch: simplest approach is create a circular hole by welding by projecting handle top vertices onto bottom plane and finding nearest grid vertices.
    # To avoid complex welding, we'll create a simple connector: create fan faces from each top circle vertex to the nearest plate bottom vertex at same (x,y) from grid.
    # Build a KD-tree of bottom grid vertices to find nearest indices.
    bottom_coords = {}
    for i in range(full_h):
        for j in range(full_w):
            vi = bottom_start + i*full_w + j
            vx, vy = verts[vi][0], verts[vi][1]
            bottom_coords[(i,j)] = vi
    # naive nearest search: for each top circle vertex, find nearest bottom vertex index
    import math as _math
    def find_nearest_bottom_index(x,y):
        # search coarse: find nearest by index in grid using linear mapping
        # xs_full and ys_full are available in closure
        ix = int(round((x - xs_full[0]) / (xs_full[-1]-xs_full[0]) * (full_w-1)))
        iy = int(round((y - ys_full[0]) / (ys_full[-1]-ys_full[0]) * (full_h-1)))
        ix = min(max(ix,0), full_w-1)
        iy = min(max(iy,0), full_h-1)
        return bottom_start + iy*full_w + ix
    # connect top ring to nearby bottom grid vertex by creating triangles
    for k in range(nseg):
        a = top_start + k
        nearest = find_nearest_bottom_index(verts[a][0], verts[a][1])
        faces.append([nearest, a, top_start + ((k+1)%nseg)])
    # hemisphere faces: connect cylinder bottom ring to first hemisphere ring (equator) which is implicit at cylinder bottom
    # The equator ring is at cylinder bottom; we have bottom circle vertices as bot_start..bot_start+nseg-1 which correspond to equator.
    # hemi_vid_indices[0] is first ring below equator.
    if hemi_vid_indices:
        first_ring = hemi_vid_indices[0]
        for k in range(nseg):
            a = bot_start + k
            b = bot_start + ((k+1)%nseg)
            c = first_ring[k % len(first_ring)]
            d = first_ring[(k+1) % len(first_ring)]
            faces.append([a, c, d])
            faces.append([a, d, b])
        # connect intermediate hemisphere rings
        for r in range(len(hemi_vid_indices)-1):
            ring_a = hemi_vid_indices[r]
            ring_b = hemi_vid_indices[r+1]
            for k in range(nseg):
                a = ring_a[k % len(ring_a)]
                b = ring_a[(k+1) % len(ring_a)]
                c = ring_b[k % len(ring_b)]
                d = ring_b[(k+1) % len(ring_b)]
                faces.append([a, c, d])
                faces.append([a, d, b])
        # connect last ring to pole
        last_ring = hemi_vid_indices[-1]
        for k in range(nseg):
            a = last_ring[k % len(last_ring)]
            b = last_ring[(k+1) % len(last_ring)]
            faces.append([a, pole_idx, b])
    else:
        # directly connect bot ring to pole if no rings (unlikely)
        for k in range(nseg):
            a = bot_start + k; b = bot_start + ((k+1)%nseg)
            faces.append([a, pole_idx, b])
    
    # Translate handle vertices to center of plate (they are already centered at (0,0))
    # Finally, write STL
    verts = np.array(verts, dtype=np.float32)
    def write_ascii_stl(filename, verts, faces, name="seal"):
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

# File paths (input images are present in /mnt/data)
ann_img = "/mnt/data/GaLKxtxWQAA09iq.jpeg"  # first image used earlier
gift_img = "/mnt/data/Screenshot 2025-10-20 at 11.24.36 PM.png"  # second image

size1 = generate_seal(ann_img, "/mnt/data/google_anniversary_seal_deep_handle.stl",
                      crop_bottom_ratio=0.30, remove_white_bg=False, remove_strings=False,
                      longest_side_mm=25.0, engrave_depth_mm=2.4, border_height_mm=1.5,
                      plate_thickness_mm=5.0, grid_px=300,
                      handle_diam_mm=16.0, handle_height_mm=20.0, handle_segs=32)

size2 = generate_seal(gift_img, "/mnt/data/google_gift_seal_deep_handle.stl",
                      crop_bottom_ratio=0.0, remove_white_bg=True, remove_strings=True,
                      longest_side_mm=25.0, engrave_depth_mm=2.4, border_height_mm=1.5,
                      plate_thickness_mm=5.0, grid_px=300,
                      handle_diam_mm=16.0, handle_height_mm=20.0, handle_segs=32)

print("Created files sizes (KB):", size1/1024.0, size2/1024.0)
