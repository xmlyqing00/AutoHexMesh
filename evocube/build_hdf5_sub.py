# -*- coding: utf-8 -*-
import h5py
import numpy as np
import argparse
import meshio
from pathlib import Path
import open3d as o3d
import os

def get_mesh_center_and_scale(points):
    """
    ????????????
    """
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    center = (min_bound + max_bound) / 2.0
    scale = (max_bound - min_bound).max()
    return center, scale

def apply_transform(points, src_center, src_scale, dst_center, dst_scale):
    """
    ??????: (P - src_c) * (dst_s / src_s) + dst_c
    """
    if src_scale < 1e-6: src_scale = 1.0
    scale_factor = dst_scale / src_scale
    return (points - src_center) * scale_factor + dst_center

def make_voxels_solid(voxel_grid):
    """
    [Flood Fill] ???????
    """
    voxels = voxel_grid.get_voxels()
    if not voxels:
        return set(), voxel_grid.origin

    indices = np.array([v.grid_index for v in voxels])
    min_idx = indices.min(axis=0)
    max_idx = indices.max(axis=0)
    dims = max_idx - min_idx + 1
    
    pad = 2
    dense_shape = dims + 2 * pad
    dense_grid = np.zeros(dense_shape, dtype=bool)
    
    shifted_indices = indices - min_idx + pad
    dense_grid[shifted_indices[:, 0], shifted_indices[:, 1], shifted_indices[:, 2]] = True
    
    # BFS Flood Fill
    queue = [(0, 0, 0)]
    visited = np.zeros(dense_shape, dtype=bool)
    visited[0, 0, 0] = True
    
    D, H, W = dense_shape
    neighbors = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
    
    while queue:
        cx, cy, cz = queue.pop(0)
        for dx, dy, dz in neighbors:
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            if 0 <= nx < D and 0 <= ny < H and 0 <= nz < W:
                if not visited[nx, ny, nz]:
                    if not dense_grid[nx, ny, nz]:
                        visited[nx, ny, nz] = True
                        queue.append((nx, ny, nz))
    
    solid_mask = ~visited
    solid_indices_dense = np.argwhere(solid_mask)
    
    final_indices_set = set()
    for idx in solid_indices_dense:
        original_idx = idx - pad + min_idx
        final_indices_set.add(tuple(original_idx))
        
    return final_indices_set, voxel_grid.origin

if __name__ == '__main__':
    parser = argparse.ArgumentParser('hdf5 reader 2-step transform')
    parser.add_argument('--dir', type=str, required=True, help='Current (Negative) output directory')
    parser.add_argument('--subtract_dir', type=str, default=None, help='Directory of the clean mesh to subtract')
    parser.add_argument('--res', type=float, default=0.1, help='cuboid resolution')
    args = parser.parse_args()

    input_dir = Path(args.dir)
    neg_input_path = input_dir / 'input.obj' 
    neg_poly_path = input_dir / 'fast_polycube_surf.obj'
    tet_path = input_dir / 'tetra.mesh'

    print(f"--- Processing: {input_dir} ---")
    
    if not (neg_input_path.exists() and neg_poly_path.exists() and tet_path.exists()):
        print(f"Error: Required files not found in {input_dir}")
        exit(1)

    # 1. ?? Negative Side ??
    neg_input_mesh = meshio.read(str(neg_input_path)) 
    neg_poly_mesh = meshio.read(str(neg_poly_path))   
    tet_mesh = meshio.read(str(tet_path))             

    # 2. ????????? (???? HDF5 ??)
    tet_v = tet_mesh.points
    tet_cells = tet_mesh.cells_dict['tetra'].astype(np.int32)
    v_min, v_max = tet_v.min(), tet_v.max()
    tet_v_norm = 2 * (tet_v - v_min) / (v_max - v_min) - 1
    global_scale = 2.0 / (v_max - v_min)

    # 3. ?? Negative Polycube (Box)
    # ??? Box
    polycube_v_scaled = neg_poly_mesh.points * global_scale
    offset = polycube_v_scaled.mean(axis=0) - tet_v_norm.mean(axis=0)
    polycube_v_final = polycube_v_scaled - offset
    
    # ?? Main O3D Mesh
    polycube_o3d = o3d.geometry.TriangleMesh()
    polycube_o3d.vertices = o3d.utility.Vector3dVector(polycube_v_final)
    polycube_o3d.triangles = o3d.utility.Vector3iVector(neg_poly_mesh.cells_dict['triangle'])

    # ??? Box
    print("--- Voxelizing Main Mesh (Solidifying)... ---")
    voxel_size = args.res
    main_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(polycube_o3d, voxel_size)
    main_solid_indices, origin = make_voxels_solid(main_grid)
    print(f"   > Main Grid Solid Voxels: {len(main_solid_indices)}")

    # 4. ???? (????)
    indices_to_subtract = set()
    
    if args.subtract_dir:
        sub_dir_path = Path(args.subtract_dir)
        # ?????????
        sub_poly_path = sub_dir_path / 'fast_polycube_surf.obj'
        sub_input_path = sub_dir_path / 'input.obj' # ???????????
        
        if sub_poly_path.exists() and sub_input_path.exists():
            print(f"--- Found Subtraction Polycube: {sub_poly_path} ---")
            plane_poly_mesh = meshio.read(str(sub_poly_path))
            plane_input_mesh = meshio.read(str(sub_input_path))
            
            # === ????????? ===
            # A. ???? (Source)
            plane_poly_c, plane_poly_s = get_mesh_center_and_scale(plane_poly_mesh.points)
            plane_input_c, plane_input_s = get_mesh_center_and_scale(plane_input_mesh.points)
            
            # B. ???? (Target)
            neg_input_c, neg_input_s = get_mesh_center_and_scale(neg_input_mesh.points)
            neg_poly_c, neg_poly_s = get_mesh_center_and_scale(neg_poly_mesh.points)

            print("--- Calculating 2-Step Transformation ---")
            
            # === ???: Plane Polycube -> Plane Input (???????) ===
            # ??: (PlanePoly - PolyC) * (InputS / PolyS) + InputC
            print(f"   > Step 1: Un-normalize Plane (Poly -> Input)")
            pts_world = apply_transform(
                plane_poly_mesh.points, 
                src_center=plane_poly_c, src_scale=plane_poly_s,
                dst_center=plane_input_c, dst_scale=plane_input_s
            )
            
            # === ???: Plane Input (World) -> Negative Polycube (??????) ===
            # ??: (WorldPts - NegInputC) * (NegPolyS / NegInputS) + NegPolyC
            # ??: Plane Input ? Negative Input ????? World ??? (? input.obj ??????)
            print(f"   > Step 2: Re-normalize to Box (Input -> Poly)")
            pts_in_neg_poly_space = apply_transform(
                pts_world,
                src_center=neg_input_c, src_scale=neg_input_s,
                dst_center=neg_poly_c, dst_scale=neg_poly_s
            )

            # === ???: ????? (Apply Global EvoCube Normalization) ===
            # ???????? tetra.mesh ? bounds,? Box ??????????
            clean_pts_norm = pts_in_neg_poly_space * global_scale
            clean_pts_final = clean_pts_norm - offset
            
            # ?? Plane O3D Mesh
            clean_o3d = o3d.geometry.TriangleMesh()
            clean_o3d.vertices = o3d.utility.Vector3dVector(clean_pts_final)
            if 'triangle' in plane_poly_mesh.cells_dict:
                clean_o3d.triangles = o3d.utility.Vector3iVector(plane_poly_mesh.cells_dict['triangle'])
            
            # ??? Plane ???
            print("--- Voxelizing Plane (Solidifying)... ---")
            sub_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(clean_o3d, voxel_size)
            
            sub_solid_indices, sub_origin = make_voxels_solid(sub_grid)
            print(f"   > Plane Solid Voxels: {len(sub_solid_indices)}")
            
            print(f"   > Mapping to Main Grid frame...")
            for idx in sub_solid_indices:
                pos_world = sub_origin + np.array(idx) * voxel_size
                idx_in_main = np.round((pos_world - origin) / voxel_size).astype(int)
                indices_to_subtract.add(tuple(idx_in_main))
                
            print(f"--- Subtraction Set Prepared: {len(indices_to_subtract)} voxels ---")
        else:
             print(f"Warning: Required plane files (input.obj, fast_polycube_surf.obj) not found in {sub_dir_path}")

    # 5. ????
    final_cuboids = []
    halflength = np.array([voxel_size / 2, voxel_size / 2, voxel_size / 2])
    
    removed_count = 0
    kept_count = 0
    
    for idx in main_solid_indices:
        if idx in indices_to_subtract:
            removed_count += 1
            continue
            
        pos = np.array(idx) * voxel_size + origin
        cuboid = np.concatenate([halflength, pos])
        final_cuboids.append(cuboid)
        kept_count += 1

    print(f"Solid Main: {len(main_solid_indices)}, Removed: {removed_count}, Final: {kept_count}")

    # 6. Save HDF5
    if len(final_cuboids) == 0:
         print("Error: No cuboids left!")
         cuboids = np.zeros((6, 0), dtype=np.float32)
         polycube_num = 0
    else:
         cuboids = np.array(final_cuboids).astype(np.float32).transpose()
         polycube_num = cuboids.shape[1]

    hdf5_path = input_dir / 'evocube.hdf5'
    print(f"Writing to {hdf5_path} ...")
    
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        g = hdf5_file.create_group('target_volume_mesh')
        g.create_dataset('vertices', data=tet_v_norm.transpose())
        g.create_dataset('tets', data=tet_cells.transpose())

        g = hdf5_file.create_group('deformed_volume_mesh')
        g.create_dataset('vertices', data=tet_v_norm.transpose())
        g.create_dataset('tets', data=tet_cells.transpose())
        
        g = hdf5_file.create_group('polycube_info')
        g.create_dataset('ordering', data=np.arange(polycube_num, dtype=np.int32))
        g.create_dataset('locked', data=np.zeros(polycube_num, dtype=np.int32))
        names = np.zeros((polycube_num, 32), dtype=np.uint8)
        for i in range(polycube_num):
            name = f'Cuboid_{i:04d}'
            names[i, :len(name)] = np.array([ord(c) for c in name], dtype=np.uint8)
        g.create_dataset('names', data=names)

        g = hdf5_file.create_group('polycube')
        g.create_dataset('params', data=cuboids)

    print("Done.")