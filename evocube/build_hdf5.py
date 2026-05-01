import h5py
import numpy as np
import argparse
import meshio
from pathlib import Path
import open3d as o3d


if __name__ == '__main__':

    parser = argparse.ArgumentParser('hdf5 reader')
    parser.add_argument('--dir', type=str, default='/space/output/toy_plane', help='hdf5 file to read')
    parser.add_argument('--res', type=float, default=0.1, help='cuboid resolution')
    args = parser.parse_args()

    # f = h5py.File('/home/zhisheng/lyq/evocube/output/plane_67_trimesh.hdf5', 'r')
    # f = h5py.File('/home/zhisheng/lyq/evocube/output/raw_obj/plane_67_trimesh/evocube.hdf5', 'r')
    # print(f.keys())

    # pts = np.array(f['target_volume_mesh']['vertices'])
    # print(pts.mean())
    # 'polycube': 'params', 6 x 2
    # 'target_volume_mesh': tets (4 x f), vertices (4 x n)
    # 'deformed_volume_mesh': tets (4 x f), vertices (4 x n)
    # 'polycube_info': ['locked', 'names', 'ordering']
    #      ordering: N
    #      locked: N
    #      names: N
    # print(f['polycube'])

    # combine the array of char to string, then decode it
    # name = ''.join([chr(i) for i in f['polycube_info']['names'][0]])
    # np.array(f['polycube_info']['names'][0])


    input_dir = Path(args.dir)
    tet_path = input_dir / 'tetra.mesh'
    polycube_path = input_dir / 'fast_polycube_surf.obj'
    polycube = meshio.read(str(polycube_path))
    print("Points:", polycube.points.shape)
    print("Cells:", polycube.cells_dict['triangle'].shape)

    mesh = meshio.read(str(tet_path))
    print("Points:", mesh.points.shape)
    print("Cells:", mesh.cells_dict['tetra'].shape)

    v = mesh.points
    tetra = mesh.cells_dict['tetra'].astype(np.int32)

    # normalize vertices to -1 ~ 1
    v_min, v_max = v.min(), v.max()
    v = 2 * (v - v_min) / (v_max - v_min) - 1

    # normalize polycube
    polycube_v = polycube.points * 2 / (v_max - v_min)
    offset = polycube_v.mean(axis=0) - v.mean(axis=0)
    polycube_v = polycube_v - offset

    polycude_o3d = o3d.geometry.TriangleMesh()
    polycude_o3d.vertices = o3d.utility.Vector3dVector(polycube_v)
    polycude_o3d.triangles = o3d.utility.Vector3iVector(polycube.cells_dict['triangle'])
    # o3d.visualization.draw_geometries([polycude_o3d])

    voxel_size = args.res
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(polycude_o3d, voxel_size)

     # o3d.visualization.draw_geometries([voxel_grid])
    # convert voxel_grid to list of cuboids
    # Retrieve voxel positions
    cuboids = []
    halflength = np.array([voxel_size / 2, voxel_size / 2, voxel_size / 2])
    for voxel in voxel_grid.get_voxels():
        pos = np.array(voxel.grid_index) * voxel_size + voxel_grid.origin
        cuboid = np.concatenate([halflength, pos])
        cuboids.append(cuboid)
    cuboids = np.array(cuboids).astype(np.float32).transpose()
    polycube_num = cuboids.shape[1]
    print("Cuboids:", cuboids.shape)
        
    # write to hdf5
    hdf5_file = h5py.File(input_dir / 'evocube.hdf5', 'w')

    # write target_volume_mesh
    g = hdf5_file.create_group('target_volume_mesh')
    g.create_dataset('vertices', data=v.transpose())
    g.create_dataset('tets', data=tetra.transpose())

    # write deformed_volume_mesh
    g = hdf5_file.create_group('deformed_volume_mesh')
    g.create_dataset('vertices', data=v.transpose())
    g.create_dataset('tets', data=tetra.transpose())
    
    # write polycube_info
    g = hdf5_file.create_group('polycube_info')
    g.create_dataset('ordering', data=np.arange(polycube_num, dtype=np.int32))
    g.create_dataset('locked', data=np.zeros(polycube_num, dtype=np.int32))
    names = np.zeros((polycube_num, 32), dtype=np.uint8)
    for i in range(polycube_num):
        name = f'Cuboid_{i:04d}'
        names[i, :len(name)] = np.array([ord(c) for c in name], dtype=np.uint8)
    g.create_dataset('names', data=names)

    # write polycube
    g = hdf5_file.create_group('polycube')
    
    # first cuboid
    # halflength = [
    #     (polycube_v[:, 0].max() - polycube_v[:, 0].min()) / 2,
    #     (polycube_v[:, 1].max() - polycube_v[:, 1].min()) / 2,
    #     (polycube_v[:, 2].max() - polycube_v[:, 2].min()) / 2
    # ]
    # center = polycube_v.mean(axis=0)
    # cuboid.append(np.concat([halflength, center]))

    # for i in range(0, polycube_num):
    #     halflength = [0.1, 0.2, 0.3]
    #     center = np.random.rand(3) * 2 - 1
    #     cuboids.append(np.concat([halflength, center]))
    # cuboids = np.array(cuboids).transpose().astype(np.float32)
    g.create_dataset('params', data=cuboids)

    hdf5_file.close()

