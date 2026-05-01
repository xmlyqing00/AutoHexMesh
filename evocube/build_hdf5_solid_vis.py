# -*- coding: utf-8 -*-
import h5py
import numpy as np
import argparse
import meshio
from pathlib import Path
import open3d as o3d
import os

# ==========================================
# Part 1: Solid Voxelization Logic
# ==========================================

def make_voxels_solid(voxel_grid):
    """
    ?? Flood Fill ??????????,???????
    ??: (set of indices tuples, origin)
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
    
    # BFS Flood Fill from corner (assuming corner is outside)
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
    
    # Invert visited to get solid interior
    solid_mask = ~visited
    solid_indices_dense = np.argwhere(solid_mask)
    
    final_indices_set = set()
    for idx in solid_indices_dense:
        original_idx = idx - pad + min_idx
        final_indices_set.add(tuple(original_idx))
        
    return final_indices_set, voxel_grid.origin

# ==========================================
# Part 2: Merging Logic (New Feature)
# ==========================================

def merge_voxels_greedy(indices_set, voxel_size, origin):
    """
    ???????????,?????????????
    
    Returns:
        merged_cuboids: List of numpy arrays, each [hx, hy, hz, cx, cy, cz]
    """
    if not indices_set:
        return []

    indices = np.array(list(indices_set))
    min_idx = indices.min(axis=0)
    max_idx = indices.max(axis=0)
    dims = max_idx - min_idx + 1

    # 1. ??????
    grid = np.zeros(dims, dtype=bool)
    # ? set ????? grid
    shifted_indices = indices - min_idx
    grid[shifted_indices[:, 0], shifted_indices[:, 1], shifted_indices[:, 2]] = True

    merged_cuboids = []
    
    D, H, W = dims

    # 2. ??????
    # ??:????? True ?,? X ????,?? Y ????,?? Z ????
    for x in range(D):
        for y in range(H):
            for z in range(W):
                if grid[x, y, z]:
                    # ????? (x, y, z)
                    
                    # --- Step A: ?? X ---
                    dx = 1
                    while (x + dx < D) and grid[x + dx, y, z]:
                        dx += 1
                    
                    # --- Step B: ?? Y ---
                    # ????????? row x:x+dx ? Y ??
                    dy = 1
                    while (y + dy < H):
                        # ???? grid[x:x+dx, y+dy, z] ???? True
                        if np.all(grid[x : x + dx, y + dy, z]):
                            dy += 1
                        else:
                            break
                    
                    # --- Step C: ?? Z ---
                    # ?????????? x:x+dx, y:y+dy ? Z ??
                    dz = 1
                    while (z + dz < W):
                        # ???? grid[x:x+dx, y:y+dy, z+dz] ???? True
                        if np.all(grid[x : x + dx, y : y + dy, z + dz]):
                            dz += 1
                        else:
                            break
                    
                    # 3. ????????????
                    grid[x : x + dx, y : y + dy, z : z + dz] = False
                    
                    # 4. ??????
                    # ?? grid ??????????: start_idx + min_idx
                    start_grid_idx = np.array([x, y, z])
                    real_start_idx = start_grid_idx + min_idx
                    
                    size_grid = np.array([dx, dy, dz]) # ???????
                    
                    # ??????????
                    # ???????: origin + real_start_idx * voxel_size
                    # ????????: origin + (real_start_idx + size_grid - 1) * voxel_size
                    # ???? = (First + Last) / 2
                    
                    first_voxel_pos = origin + real_start_idx * voxel_size
                    last_voxel_pos = origin + (real_start_idx + size_grid - 1) * voxel_size
                    
                    center = (first_voxel_pos + last_voxel_pos) / 2.0
                    
                    # ??? = (???? * voxel_size) / 2
                    halflength = (size_grid * voxel_size) / 2.0
                    
                    # ??: [hx, hy, hz, cx, cy, cz]
                    cuboid_params = np.concatenate([halflength, center])
                    merged_cuboids.append(cuboid_params)

    return merged_cuboids

# ==========================================
# Part 3: Visualization Logic
# ==========================================

def convert_cubes_to_hexes(cubes):
    """
    ????????? VTK ???????????
    cubes shape: (N, 6) -> [hx, hy, hz, cx, cy, cz]
    """
    if cubes is None or len(cubes) == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 8), dtype=np.int64)

    corner_offsets = np.array(
        [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
         (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
        dtype=float,
    )

    vertex_chunks = []
    connectivity_chunks = []
    start_index = 0

    for cube in cubes:
        cube_values = np.asarray(cube, dtype=float).ravel()
        half_extents = cube_values[:3]
        center = cube_values[3:]
        vertices = center + corner_offsets * half_extents

        vertex_chunks.append(vertices)
        connectivity_chunks.append(
            np.arange(start_index, start_index + 8, dtype=np.int64)
        )
        start_index += 8

    all_vertices = np.vstack(vertex_chunks) if vertex_chunks else np.zeros((0, 3), dtype=float)
    all_connectivity = np.vstack(connectivity_chunks) if connectivity_chunks else np.zeros((0, 8), dtype=np.int64)

    return all_vertices, all_connectivity

def write_hex_mesh(vertices, connectivity, vtk_filename):
    meshio.write(
        vtk_filename,
        meshio.Mesh(points=vertices, cells=[("hexahedron", connectivity)]),
    )

# ==========================================
# Part 4: Main Build Process
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser('HDF5 Builder with Merged Cuboids')
    parser.add_argument('--dir', type=str, required=True, help='Input/Output directory')
    parser.add_argument('--res', type=float, default=0.1, help='Cuboid resolution')
    args = parser.parse_args()

    input_dir = Path(args.dir)
    neg_poly_path = input_dir / 'fast_polycube_surf.obj'
    tet_path = input_dir / 'tetra.mesh'

    print(f"--- Processing: {input_dir} ---")
    
    if not (neg_poly_path.exists() and tet_path.exists()):
        print(f"Error: Required files (fast_polycube_surf.obj, tetra.mesh) not found in {input_dir}")
        exit(1)

    # 1. ????
    poly_mesh = meshio.read(str(neg_poly_path))   
    tet_mesh = meshio.read(str(tet_path))             

    # 2. ????? (?? tetra mesh)
    tet_v = tet_mesh.points
    tet_cells = tet_mesh.cells_dict['tetra'].astype(np.int32)
    v_min, v_max = tet_v.min(), tet_v.max()
    
    # ???? [-1, 1]
    tet_v_norm = 2 * (tet_v - v_min) / (v_max - v_min) - 1
    global_scale = 2.0 / (v_max - v_min)

    # 3. ?? Polycube
    polycube_v_scaled = poly_mesh.points * global_scale
    offset = polycube_v_scaled.mean(axis=0) - tet_v_norm.mean(axis=0)
    polycube_v_final = polycube_v_scaled - offset
    
    # ?? Open3D ????
    polycube_o3d = o3d.geometry.TriangleMesh()
    polycube_o3d.vertices = o3d.utility.Vector3dVector(polycube_v_final)
    if 'triangle' in poly_mesh.cells_dict:
        polycube_o3d.triangles = o3d.utility.Vector3iVector(poly_mesh.cells_dict['triangle'])

    # 4. ???????
    print(f"--- Voxelizing (Resolution: {args.res})... ---")
    voxel_size = args.res
    
    # Open3D ???????
    surface_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(polycube_o3d, voxel_size)
    
    # ?? Flood Fill ????
    solid_indices, origin = make_voxels_solid(surface_grid)
    raw_voxel_count = len(solid_indices)
    print(f"   > Raw Solid Voxels: {raw_voxel_count}")

    # 5. ???? (Merge)
    print("--- Merging Voxels into Maximal Cuboids... ---")
    final_cuboids_list = merge_voxels_greedy(solid_indices, voxel_size, origin)
    merged_count = len(final_cuboids_list)
    print(f"   > Merged into {merged_count} cuboids (Compression: {100*(1 - merged_count/raw_voxel_count):.1f}%)")

    if merged_count == 0:
        print("Error: No voxels generated!")
        cuboids_array = np.zeros((6, 0), dtype=np.float32)
        polycube_num = 0
    else:
        # HDF5 ????????? (6, N)
        cuboids_array = np.array(final_cuboids_list).astype(np.float32).transpose()
        polycube_num = cuboids_array.shape[1]

    # 6. ?? HDF5 (?? Merge ????)
    hdf5_path = input_dir / 'evocube.hdf5'
    print(f"Writing HDF5 to {hdf5_path} ...")
    
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
        g.create_dataset('params', data=cuboids_array)

    # 7. ????????? (Raw vs Merged)
    print("Writing visualizations...")
    
    # A. ?????? (? HDF5 ??)
    if merged_count > 0:
        vis_path_merged = input_dir / 'cuboids_view_merged.vtu'
        # ??? (N, 6) ? meshio ??
        cubes_params = cuboids_array.transpose()
        vertices, connectivity = convert_cubes_to_hexes(cubes_params)
        write_hex_mesh(vertices, connectivity, str(vis_path_merged))
        print(f"   > Saved merged view: {vis_path_merged}")
        
    # B. ?????? (????,????)
    if raw_voxel_count > 0:
        vis_path_raw = input_dir / 'cuboids_view_raw.vtu'
        
        # ???? raw cuboids list
        raw_cuboids = []
        raw_halflength = np.array([voxel_size / 2, voxel_size / 2, voxel_size / 2])
        for idx in solid_indices:
            pos = np.array(idx) * voxel_size + origin
            raw_cuboids.append(np.concatenate([raw_halflength, pos]))
            
        raw_cuboids = np.array(raw_cuboids) # (N, 6)
        
        vertices, connectivity = convert_cubes_to_hexes(raw_cuboids)
        write_hex_mesh(vertices, connectivity, str(vis_path_raw))
        print(f"   > Saved raw voxel view: {vis_path_raw}")
    
    print("Done.")