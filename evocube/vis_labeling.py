# -*- coding: utf-8 -*-
import numpy as np
import argparse
import meshio
from pathlib import Path

# ==========================================
# ???? (RGB 0-255)
# ==========================================
COLORS = np.array([
    [255, 0, 0],    # 0: +X (Red)
    [0, 255, 255],  # 1: -X (Cyan)
    [0, 255, 0],    # 2: +Y (Green)
    [255, 0, 255],  # 3: -Y (Magenta)
    [0, 0, 255],    # 4: +Z (Blue)
    [255, 255, 0],  # 5: -Z (Yellow)
], dtype=int)

GREY = [200, 200, 200]

def write_ply_manual(output_path, vertices, faces, labels):
    """
    ???? ASCII PLY ??,?? MeshLab ?????????
    """
    num_verts = len(vertices)
    num_faces = len(faces)
    
    print(f"Writing PLY to {output_path}...")
    
    with open(output_path, 'w') as f:
        # --- Header ---
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_verts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {num_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        # ??????? (MeshLab ????)
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # --- Vertices ---
        # ??????
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
        # --- Faces & Colors ---
        # ??: 3 v1 v2 v3 r g b
        for i, face in enumerate(faces):
            lbl = labels[i]
            if 0 <= lbl < 6:
                c = COLORS[lbl]
            else:
                c = GREY
            
            f.write(f"3 {face[0]} {face[1]} {face[2]} {c[0]} {c[1]} {c[2]}\n")

def process(input_dir):
    input_path = Path(input_dir)
    mesh_path = input_path / 'boundary.obj'
    label_path = input_path / 'labeling.txt'
    output_path = input_path / 'labeled_mesh.ply' # ?? PLY

    if not mesh_path.exists() or not label_path.exists():
        print(f"Error: Files not found in {input_dir}")
        return

    # 1. ????
    mesh = meshio.read(str(mesh_path))
    if 'triangle' not in mesh.cells_dict:
        print("Error: Mesh must be triangle mesh.")
        return
    
    vertices = mesh.points
    faces = mesh.cells_dict['triangle']

    # 2. ????
    try:
        labels = np.loadtxt(str(label_path), dtype=int)
    except:
        print("Error reading labeling.txt")
        return

    # 3. ????
    if len(labels) != len(faces):
        print(f"Warning: Labels {len(labels)} != Faces {len(faces)}. Adjusting.")
        if len(labels) > len(faces):
            labels = labels[:len(faces)]
        else:
            labels = np.pad(labels, (0, len(faces) - len(labels)), constant_values=-1)

    # 4. ???
    write_ply_manual(output_path, vertices, faces, labels)
    print("Done. Generated 'labeled_mesh.ply'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()
    process(args.dir)