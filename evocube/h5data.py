import argparse
import os
import h5py
import numpy as np
import meshio
from types import SimpleNamespace


def convert_cubes_to_hexes(cubes):
    """
    Convert a list of axis-aligned cubes into VTK hexahedral vertices/connectivity.

    Parameters
    ----------
    cubes : iterable of sequences
        Each cube must be defined by six numbers (hx, hy, hz, cx, cy, cz) where
        (cx, cy, cz) is the center and (hx, hy, hz) are the half-widths.

    Returns
    -------
    vertices : ndarray, shape (N, 3)
        All cube vertices (duplicates allowed).
    connectivity : ndarray, shape (M, 8)
        VTK-conformant hexahedron connectivity using 0-based indexing.
    """
    if cubes is None:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 8), dtype=np.int64)

    corner_offsets = np.array(
        [
            (-1, -1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
        ],
        dtype=float,
    )

    vertex_chunks = []
    connectivity_chunks = []
    start_index = 0

    for cube in cubes:
        cube_values = np.asarray(cube, dtype=float).ravel()
        if cube_values.size != 6:
            raise ValueError("Each cube must contain exactly six values (hx, hy, hz, cx, cy, cz).")

        half_extents = cube_values[:3]
        center = cube_values[3:]
        vertices = center + corner_offsets * half_extents

        vertex_chunks.append(vertices)
        connectivity_chunks.append(
            np.arange(start_index, start_index + 8, dtype=np.int64)
        )
        start_index += 8

    all_vertices = np.vstack(vertex_chunks) if vertex_chunks else np.zeros((0, 3), dtype=float)
    all_connectivity = (
        np.vstack(connectivity_chunks) if connectivity_chunks else np.zeros((0, 8), dtype=np.int64)
    )

    return all_vertices, all_connectivity


def write_hex_mesh(vertices, connectivity, vtk_filename):
    """
    Export hexahedral mesh defined by vertices/connectivity to VTU via meshio.

    Parameters
    ----------
    vertices : array_like
        Point coordinates with shape (N, 3).
    connectivity : array_like
        Hex connectivity with shape (M, 8) following VTK ordering.
    vtk_filename : str
        Output VTU filename.
    """
    vertices = np.asarray(vertices, dtype=float)
    connectivity = np.asarray(connectivity, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3).")
    if connectivity.ndim != 2 or connectivity.shape[1] != 8:
        raise ValueError("connectivity must have shape (M, 8).")

    meshio.write(
        vtk_filename,
        meshio.Mesh(points=vertices, cells=[("hexahedron", connectivity)]),
    )


class MyData:
    def __init__(self, input_filename=None):
        """
        Create a MyData object.
        If input_filename is provided and exists, load all data immediately.
        Otherwise create an empty data container.
        """
        self.input_filename = input_filename
        self.data = SimpleNamespace()

        if input_filename is None or not os.path.isfile(input_filename):
            print(f"Input HDF5 file '{input_filename}' not found. Creating empty MyData object.")
            return

        self._load_from_h5(input_filename)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def _load_from_h5(self, filename):
        """Load all groups/datasets from HDF5 into self.data as NumPy arrays."""
        with h5py.File(filename, "r") as f:
            for group_name in f.keys():
                group_obj = SimpleNamespace()
                group = f[group_name]

                for dataset_name in group.keys():
                    dset = group[dataset_name]
                    group_obj.__dict__[dataset_name] = np.array(dset)

                self.data.__dict__[group_name] = group_obj

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def display(self):
        """Print the tree-like structure of groups/datasets with dtype and shape."""
        for group_name in self.data.__dict__:
            print(f"{group_name}/")
            group_obj = self.data.__dict__[group_name]

            for var_name in group_obj.__dict__:
                arr = group_obj.__dict__[var_name]
                print(f"    {var_name}: {arr.dtype}, shape={arr.shape}")

            print()

    # ------------------------------------------------------------------
    # Add / Delete Groups
    # ------------------------------------------------------------------
    def addGroup(self, name):
        """Create an empty group."""
        if hasattr(self.data, name):
            print(f"Warning: group '{name}' already exists. Skipping creation.")
            return
        self.data.__dict__[name] = SimpleNamespace()

    def deleteGroup(self, name):
        """Remove an entire group."""
        if hasattr(self.data, name):
            del self.data.__dict__[name]
        else:
            print(f"Warning: group '{name}' does not exist. Nothing to delete.")

    # ------------------------------------------------------------------
    # Save to HDF5
    # ------------------------------------------------------------------
    def save(self, filename):
        """Write current structure to a new HDF5 file."""
        if os.path.isfile(filename):
            print(f"Warning: output file '{filename}' already exists and will be overwritten.")

        with h5py.File(filename, "w") as f:
            for group_name, group_obj in self.data.__dict__.items():
                g = f.create_group(group_name)

                for var_name, array in group_obj.__dict__.items():
                    g.create_dataset(var_name, data=array)

    # ------------------------------------------------------------------
    # Output to VTK (user will fill implementation)
    # ------------------------------------------------------------------
    def output_to_vtk(self, group_name, vtk_filename):
        """User-defined VTK export stub."""
        if not hasattr(self.data, group_name):
            print(f"Group '{group_name}' does not exist. Nothing to output.")
            return

        print(f"(Placeholder) Exporting group '{group_name}' to '{vtk_filename}'.")
        print("You will implement MeshIO logic here.")

    # ------------------------------------------------------------------
    # Get data part
    # ------------------------------------------------------------------
    def get_tet_data(self, group_name):
        """
        Return transposed vertices and tetrahedral connectivity for a group.

        Parameters
        ----------
        group_name : str
            Name of the group stored under self.data that contains `vertices` and `tets`.

        Returns
        -------
        points : ndarray
            `vertices.T` as floating points.
        tets : ndarray
            `tets.T` as int64.
        """
        mesh = getattr(self.data, group_name, None)
        if mesh is None:
            raise ValueError(f"Group '{group_name}' does not exist.")

        vertices = getattr(mesh, "vertices", None)
        tets = getattr(mesh, "tets", None)
        if vertices is None or tets is None:
            raise ValueError(f"Group '{group_name}' missing vertices or tets.")

        return np.asarray(vertices.T, dtype=float), np.asarray(tets.T, dtype=np.int64)

    def get_hex_data(self, group_name):
        """
        Return vertices and hexahedral connectivity for a group.
        """
        mesh = getattr(self.data, group_name, None)
        if mesh is None:
            raise ValueError(f"Group '{group_name}' does not exist.")

        vertices = getattr(mesh, "vertices", None)
        hexes = getattr(mesh, "hexes", None)
        if vertices is None or hexes is None:
            raise ValueError(f"Group '{group_name}' missing vertices or hexes.")

        vertices = np.asarray(vertices.T, dtype=float)
        hexes_arr = np.asarray(hexes, dtype=np.int64)
        if hexes_arr.size % 9 != 0:
            raise ValueError(f"Group '{group_name}' hexes length not divisible by 9.")

        hex_array = hexes_arr.reshape(-1, 9)[:, :8]
        vtk_order = hex_array[:, [0, 1, 3, 2, 4, 5, 7, 6]]

        return vertices, vtk_order

    def get_tri_data(self, group_name):
        """
        Return vertices and triangular face connectivity for a group.
        """
        mesh = getattr(self.data, group_name, None)
        if mesh is None:
            raise ValueError(f"Group '{group_name}' does not exist.")

        vertices = getattr(mesh, "vertices", None)
        faces = getattr(mesh, "faces", None)
        if vertices is None or faces is None:
            raise ValueError(f"Group '{group_name}' missing vertices or faces.")

        return np.asarray(vertices.T, dtype=float), np.asarray(faces.T, dtype=np.int64)

    def get_quad_data(self, group_name):
        """
        Return vertices, quad connectivity, and patch IDs for a group.
        """
        mesh = getattr(self.data, group_name, None)
        if mesh is None:
            raise ValueError(f"Group '{group_name}' does not exist.")

        vertices = getattr(mesh, "vertices", None)
        quads = getattr(mesh, "quads", None)
        patches = getattr(mesh, "patches", None)
        if vertices is None or quads is None or patches is None:
            raise ValueError(f"Group '{group_name}' missing vertices, quads, or patches.")

        return (
            np.asarray(vertices.T, dtype=float),
            np.asarray(quads.T, dtype=np.int64),
            np.asarray(patches, dtype=np.int64).ravel(),
        )

    # ------------------------------------------------------------------
    # Output anchors as VTU point cloud
    # ------------------------------------------------------------------
    def output_anchors(self, vtk_filename):
        """
        Export anchor positions as a VTU point cloud.
        Points:     self.data.deformed_volume_mesh.anchors
        Point data: {"sdf": self.data.deformed_volume_mesh.sdf} when available
        """
        if not hasattr(self.data, "deformed_volume_mesh"):
            print("deformed_volume_mesh not present; nothing to export.")
            return

        mesh = self.data.deformed_volume_mesh
        anchors = getattr(mesh, "anchors", None)

        if anchors is None:
            print("deformed_volume_mesh missing anchors; nothing to export.")
            return

        points = np.asarray(anchors.T, dtype=float)
        point_count = points.shape[0]

        sdf_array = getattr(mesh, "sdf", None)
        if sdf_array is not None:
            sdf_array = np.asarray(sdf_array, dtype=float).ravel()
            if sdf_array.shape[0] != point_count:
                print("Warning: sdf length does not match anchors; omitting point data.")
                sdf_array = None

        cells = [("vertex", np.arange(point_count, dtype=np.int64).reshape(-1, 1))]
        point_data = {"sdf": sdf_array} if sdf_array is not None else None

        meshio.write(vtk_filename, meshio.Mesh(points=points, cells=cells, point_data=point_data))

    # ------------------------------------------------------------------
    # Output deformed volume mesh to VTU
    # ------------------------------------------------------------------
    def output_deformed_volume_mesh(self, vtk_filename):
        """
        Export the deformed volume mesh to VTU using meshio.
        Points: self.data.deformed_volume_mesh.vertices
        Cells:  self.data.deformed_volume_mesh.tets (tetrahedral connectivity)
        """
        if not hasattr(self.data, "deformed_volume_mesh"):
            print("deformed_volume_mesh not present; nothing to export.")
            return

        try:
            points, cells_arr = self.get_tet_data("deformed_volume_mesh")
        except ValueError as exc:
            print(str(exc))
            return

        cells = [("tetra", cells_arr)]

        meshio.write(vtk_filename, meshio.Mesh(points=points, cells=cells))

    # ------------------------------------------------------------------
    # Output target volume tetrahedral mesh to VTU
    # ------------------------------------------------------------------
    def output_target_tet_mesh(self, vtk_filename):
        """
        Export the target volume mesh (tetrahedra) to VTU.
        """
        if not hasattr(self.data, "target_volume_mesh"):
            print("target_volume_mesh not present; nothing to export.")
            return

        try:
            points, cells_arr = self.get_tet_data("target_volume_mesh")
        except ValueError as exc:
            print(str(exc))
            return

        meshio.write(vtk_filename, meshio.Mesh(points=points, cells=[("tetra", cells_arr)]))


    # ------------------------------------------------------------------
    # Output deformed surface mesh to OBJ
    # ------------------------------------------------------------------
    def output_deformed_mesh(self, obj_filename):
        """
        Export the deformed surface mesh to OBJ using meshio.
        Points: self.data.deformed_mesh.vertices
        Faces:  self.data.deformed_mesh.faces (tri connectivity)
        """
        if not hasattr(self.data, "deformed_mesh"):
            print("deformed_mesh not present; nothing to export.")
            return

        mesh = self.data.deformed_mesh
        vertices = getattr(mesh, "vertices", None)
        faces = getattr(mesh, "faces", None)

        if vertices is None or faces is None:
            print("deformed_mesh missing vertices or faces; nothing to export.")
            return

        points = np.asarray(vertices.T, dtype=float)
        cells = [("triangle", np.asarray(faces.T, dtype=np.int64))]

        meshio.write(obj_filename, meshio.Mesh(points=points, cells=cells))

    # ------------------------------------------------------------------
    # Output target surface triangle mesh to VTU
    # ------------------------------------------------------------------
    def output_target_tri_mesh(self, vtk_filename):
        """
        Export the target surface triangle mesh to VTU.
        """
        if not hasattr(self.data, "target_mesh"):
            print("target_mesh not present; nothing to export.")
            return

        try:
            points, cells_arr = self.get_tri_data("target_mesh")
        except ValueError as exc:
            print(str(exc))
            return

        meshio.write(vtk_filename, meshio.Mesh(points=points, cells=[("triangle", cells_arr)]))

    # ------------------------------------------------------------------
    # Output cuboids to VTK
    # ------------------------------------------------------------------
    def output_cuboids(self, vtk_filename):
        """
        Export polycube cuboids as a VTU hexahedral mesh.

        Expects `self.data.polycube.params` to hold six values per cuboid.
        """
        polycube = getattr(self.data, "polycube", None)
        if polycube is None or not hasattr(polycube, "params"):
            print("polycube.params not available; cannot export cuboids.")
            return

        params = np.asarray(polycube.params, dtype=float)
        if params.ndim == 1:
            if params.size != 6:
                print("polycube.params must describe six values per cuboid.")
                return
            cubes = [params]
        elif params.ndim == 2:
            if params.shape[0] == 6:
                cubes = params.T
            else:
                print("polycube.params must have six rows or six columns.")
                return
        else:
            print("polycube.params has unexpected shape; cannot export cuboids.")
            return

        vertices, connectivity = convert_cubes_to_hexes(cubes)
        write_hex_mesh(vertices, connectivity, vtk_filename)

    # ------------------------------------------------------------------
    # Output polycubes to VTK
    # ------------------------------------------------------------------
    def output_polycube(self, vtk_filename):
        """
        Export complex polycube as hexahedral VTU mesh.

        Expects `self.data.polycube_complex.vertices` and `.hexes`.
        """
        try:
            vertices, connectivity = self.get_hex_data("polycube_complex")
        except ValueError as exc:
            print(str(exc))
            return

        write_hex_mesh(vertices, connectivity, vtk_filename)

    # ------------------------------------------------------------------
    # Output result hex mesh to VTK
    # ------------------------------------------------------------------
    def output_result_mesh(self, vtk_filename):
        """
        Export complex polycube as hexahedral VTU mesh.

        Expects `self.data.result_mesh.vertices` and `.hexes`.
        """
        try:
            vertices, connectivity = self.get_hex_data("result_mesh")
        except ValueError as exc:
            print(str(exc))
            return

        write_hex_mesh(vertices, connectivity, vtk_filename)

    # ------------------------------------------------------------------
    # Output partial result hexahedral mesh
    # ------------------------------------------------------------------
    def output_result_mesh1(self, vtk_filename):
        """
        Export a truncated portion of the result hex mesh that excludes quad faces.
        """
        try:
            _, quads, _ = self.get_quad_data("polycube_complex")
        except ValueError as exc:
            print(str(exc))
            return

        try:
            vertices, connectivity = self.get_hex_data("result_mesh")
        except ValueError as exc:
            print(str(exc))
            return

        total_hex = connectivity.shape[0]
        quad_count = quads.shape[0]
        cutoff = max(0, total_hex - quad_count)
        if cutoff == 0:
            print("No remaining hexahedra to export after removing quad count.")
            return

        write_hex_mesh(vertices, connectivity[:cutoff], vtk_filename)
       
    # ------------------------------------------------------------------
    # Output middle hex/polycube mesh to VTK
    # ------------------------------------------------------------------
    def output_target_complex(self, vtk_filename):
        """
        Export complex polycube as hexahedral VTU mesh.

        Expects `self.data.target_complex.vertices` and `.hexes`.
        """
        try:
            vertices, connectivity = self.get_hex_data("target_complex")
        except ValueError as exc:
            print(str(exc))
            return

        write_hex_mesh(vertices, connectivity, vtk_filename)

    # ------------------------------------------------------------------
    # Batch export helper
    # ------------------------------------------------------------------
    def output_all(self, base_name, directory="."):
        """
        Run every available export, prefixing filenames with `base_name`.
        """
        prefix = f"{base_name}_" if base_name else ""

        tasks = [
            (self.output_anchors, os.path.join(directory, f"{prefix}anchors.vtu")),
            (self.output_deformed_volume_mesh, os.path.join(directory, f"{prefix}deformed_volume.vtu")),
            (self.output_deformed_mesh, os.path.join(directory, f"{prefix}deformed_mesh.obj")),
            (self.output_cuboids, os.path.join(directory, f"{prefix}cuboids.vtu")),
            (self.output_polycube, os.path.join(directory, f"{prefix}polycube.vtu")),
            (self.output_result_mesh, os.path.join(directory, f"{prefix}result_mesh.vtu")),
            (self.output_result_mesh1, os.path.join(directory, f"{prefix}result_mesh_partial.vtu")),
            (self.output_target_complex, os.path.join(directory, f"{prefix}target_complex.vtu")),
            (self.output_target_tet_mesh, os.path.join(directory, f"{prefix}target_volume.vtu")),
            (self.output_target_tri_mesh, os.path.join(directory, f"{prefix}target_tri.vtu")),
        ]

        for func, path in tasks:
            try:
                func(path)
            except Exception as exc:  # keep running even if one export fails
                print(f"Failed to export via {func.__name__}: {exc}")


def main():
    """Command-line entry point that dumps data exports for an HDF5 input."""
    parser = argparse.ArgumentParser(description="Export MyData contents to VTU/OBJ files.")
    parser.add_argument("input_h5_file", help="HDF5 file containing the MyData dataset.")
    args = parser.parse_args()

    file_path = args.input_h5_file
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]

    data = MyData(file_path)
    data.display()
    data.output_all(base_name)


if __name__ == "__main__":
    main()
