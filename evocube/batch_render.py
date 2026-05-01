# -*- coding: utf-8 -*-
import open3d as o3d
import numpy as np
import argparse
import os
import glob
import re
import time

def natural_sort_key(s):
    """
    ??????,?? iter_2 ? iter_10 ??
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def batch_render(input_dir, output_dir=None, width=1920, height=1080):
    if output_dir is None:
        output_dir = os.path.join(input_dir, "renders")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. ???????
    # ?? iter_*.obj ? iter_*.ply
    files = glob.glob(os.path.join(input_dir, "iter_*.obj"))
    if not files:
        # ?????????? .ply
        files = glob.glob(os.path.join(input_dir, "iter_*.ply"))
    
    if not files:
        print(f"Error: No 'iter_*.obj' files found in {input_dir}")
        return

    # ??????????????
    files.sort(key=natural_sort_key)
    print(f"Found {len(files)} files. Starting render...")

    # 2. ??? Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=True)
    
    # ??????
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1]) # ????
    opt.mesh_show_back_face = True
    opt.light_on = True

    # 3. ??????????,??????
    first_mesh = o3d.io.read_triangle_mesh(files[0])
    first_mesh.compute_vertex_normals()
    center = first_mesh.get_center()
    
    # ???????,????????
    bbox = first_mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_max_bound() - bbox.get_min_bound()
    max_extent = max(extent)

    # 4. ?????? (?-?-?)
    # Open3D ????? Y ???
    # ?(-X), ?(+Z), ?(+Y) -> ????????,??????????
    # ??????????? Isometric ????
    
    # LookAt: ??????
    lookat_vec = center
    
    # Up: Y???
    up_vec = [0, 1, 0]
    
    # Front: ?????? (??:?????????????,?????????)
    # ????“???”??,???????? (-x, +y, +z) ??
    # ? Open3D ? set_front ?,???????????
    # ???:[-1, -1, -1] ???????????
    front_vec = [-0.5, -0.4, -0.8] 

    # Zoom: ????
    zoom_val = 0.7

    first_run = True

    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        print(f"[{i+1}/{len(files)}] Rendering {filename}...")

        # ????
        mesh = o3d.io.read_triangle_mesh(file_path)
        
        # ??:???????????
        mesh.compute_vertex_normals()
        
        # ??:?????,??????
        mesh.paint_uniform_color([0.6, 0.7, 0.9]) 

        # ?????
        vis.add_geometry(mesh)

        # ??:?????????,???????
        # ???????????
        ctr = vis.get_view_control()
        if first_run:
            ctr.set_lookat(lookat_vec)
            ctr.set_front(front_vec)
            ctr.set_up(up_vec)
            ctr.set_zoom(zoom_val)
            first_run = False
        else:
            # ???????????,?? add_geometry ????
            ctr.set_lookat(lookat_vec)
            ctr.set_front(front_vec)
            ctr.set_up(up_vec)
            ctr.set_zoom(zoom_val)

        # ?????
        vis.poll_events()
        vis.update_renderer()

        # ????
        out_name = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
        vis.capture_screen_image(out_name, do_render=True)

        # ??????,?????
        vis.remove_geometry(mesh, reset_bounding_box=False)

    vis.destroy_window()
    print(f"Done! Renders saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch render OBJ sequence with fixed camera.")
    parser.add_argument("--dir", type=str, required=True, help="Input directory containing iter_*.obj files")
    parser.add_argument("--width", type=int, default=1280, help="Image width")
    parser.add_argument("--height", type=int, default=720, help="Image height")
    
    args = parser.parse_args()
    
    batch_render(args.dir, width=args.width, height=args.height)