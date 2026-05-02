# Docker Environment for EvoCube and Interactive-All-HexMesh
Instructions on setting up the environment for EvoCube and Interactive-All-HexMesh

Authors: Yongqing Liang and Xin Li (Texas A&M University)

## Basic Docker Program and NVIDIA Support

The following instructions are based on the Ubuntu 24.04 LTS and CUDA 12.6 (nvidia-open). A valid GitHub account with git access is required.

### 1 Clone the repository
```
sudo apt install git
git clone --recursive git@github.com:xmlyqing00/AutoHexMesh.git
cd AutoHexMesh
```
### 2 Install Docker Program
```
. ./install_docker.sh
```
<details>
<summary>References</summary>

- https://docs.docker.com/engine/install/ubuntu/
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

</details>

Test Docker (Optional): It should show the GPU information in nvidia-smi.
```
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```


## Build the environment

### 1 Setup the host environment and create a Docker image

Our Docker image is based on the **CUDA 12.4** container. It should be less or equal to the version of your host GPU driver version. If not, please check the CUDA container Reference to change the version of NVIDIA container and the LibTorch. 
<details>
<summary>Key Library Versions and References</summary>

- Docker Image based on CUDA 12.4: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags 
- LibTorch 2.6.0: https://pytorch.org/get-started/locally/ 
- Vulkan-SDK 1.3.268.0: https://vulkan.lunarg.com/sdk/home#linux

</details>

In the host machine under the folder `AutoHexMesh`, run the following command to build the Docker image.
```
. ./setup.sh
```

### 2 Enter the Docker container

A detailed document of the code structure and parameters can be found in the [Document](DOCUMENT.md).

#### 2.1 Create a Docker container at the first time
In the host machine under the folder `AutoHexMesh`,
```
. ./run_docker.sh
```
Note: `source` is the bash command feature. Using `. ./compile.sh` is more general.
#### 2.2 Resume the container after the first time
In the host machine, you can attach to the container by the following command.
```
sudo docker container list -a
xhost +local:root
sudo docker start [container id]
sudo docker attach [container id]
```

### 3 Compile the code every time you enter the container
In the Docker container under the folder `/space`,
```
. ./compile.sh
```


## Run the code

### 1 Run the evocube
In the container, to automatically generate polycube in the folder `./data/examples`
```
cd /space/evocube/build
./init_from_folder
```

Or run the labeling module with GUI in the container
```
cd /space/evocube/build
./evolabel /space/data/examples/toy_plane.obj
```

### 2 Convert the polycube to cubes (HDF5 format)
In the container
```bash
cd /space/evocube/
python3 build_hdf5.py --dir /space/output/examples/toy_plane
```

#### 2.1 build_hdf5_solid_vis.py (Alternative with Solid Voxelization)

This script provides an enhanced HDF5 generation pipeline with solid voxelization and greedy cuboid merging. It uses flood-fill to create watertight solid voxels and then merges them into larger cuboids to reduce complexity.

**Required Input Files** (in the specified directory):
- `fast_polycube_surf.obj` - Polycube surface mesh
- `tetra.mesh` - Tetrahedral mesh

**Output Files**:
- `evocube.hdf5` - HDF5 file containing normalized tet mesh and merged cuboid parameters
- `cuboids_view_merged.vtu` - VTU visualization of merged cuboids (for ParaView)
- `cuboids_view_raw.vtu` - VTU visualization of raw voxels before merging

**Usage**:
```bash
cd /space/evocube/
python3 build_hdf5_solid_vis.py --dir /space/output/examples/toy_plane --res 0.1
```

**Arguments**:
- `--dir` (required): Input/output directory containing the required mesh files
- `--res` (optional, default=0.1): Voxelization resolution (smaller values = finer voxels)

#### 2.2 vis_labeling.py (Labeling Visualization)

This script visualizes the labeling results by coloring each triangle face according to its assigned axis direction. The output is a PLY file that can be viewed in MeshLab or other 3D viewers.

**Color Mapping**:
| Label | Direction | Color   |
|-------|-----------|---------|
| 0     | +X        | Red     |
| 1     | -X        | Cyan    |
| 2     | +Y        | Green   |
| 3     | -Y        | Magenta |
| 4     | +Z        | Blue    |
| 5     | -Z        | Yellow  |

**Required Input Files** (in the specified directory):
- `boundary.obj` - Triangle surface mesh
- `labeling.txt` - Text file with one label (0-5) per face

**Output File**:
- `labeled_mesh.ply` - Colored PLY mesh for visualization

**Usage**:
```bash
cd /space/evocube/
python3 vis_labeling.py --dir /space/output/examples/toy_plane
```

The generated `labeled_mesh.ply` can be opened directly in MeshLab to inspect the labeling quality.

### 3 Run the interactive-hex-meshing
In the container
```
source /space/lib/vulkan-sdk-1.3.268.0/setup-env.sh
cd /space/interactive-hex-meshing/bin/Release
./hex
```

<details>
<summary>Tips for Vulkan loading errors.</summary>
Vulkan error is common in the Docker container. Please check the following tips to quickly solve the problems.
- Source the vulkan sdk environment in the container. `source /space/lib/vulkan-sdk-1.3.268.0/setup-env.sh
- `vulkaninfo --summary` in the container should return basic profile. There should be no error messages in the beginning lines about ICD, drivers or loading issues.
- If there is a ICD error, try to move unrelated ICD json files in `/usr/share/vulkan/icd.d/` to other folders. In my cases, I moved `nouveau_icd.json` and `intel_icd.json` to other folders, while only keep `nvidia_icd.json` in the folder.
- Check the docker run command to make sure the Vulkan library is properly loaded by mapping.

</details>

In the pop-up GUI, on the top left, load the HDF5 file generated by the last step.
Follow the instructions of the pipeline to generate the hex mesh.

1. `File` -> `Open` -> `/space/output/examples/toy_plane/evocube.hdf5`
2. `Decomposition` -> `Create anchors ...` -> `Reoptimization`
3. `Discretization` -> `Discretize` -> `Finalize polycube`
4. `Hexhedralization` -> `Init/Finalize hex mesh`

Step by step visualization:
![](./assets/stepbystep.jpg)
## Clean the Docker environment
In the host machine, remove all containers
```
sudo docker rm -f $(sudo docker ps -aq)
```
In the host machine, remove all images
```
sudo docker rmi -f $(sudo docker images -q)
```

## Copyright

Copyright (c) 2025 Yongqing Liang and Xin Li

<details>
<summary>Citations</summary>

```
@software{Liang_Docker_Environment_for_2025,
    author = {Liang, Yongqing and Li, Xin},
    month = may,
    title = {{Docker Environment for EvoCube and Interactive-All-HexMesh}},
    url = {https://github.com/xmlyqing00/AutoHexMesh},
    version = {1.0},
    year = {2025}
}

@article{dumery:evocube,
   title = {{Evocube: a Genetic Labeling Framework for Polycube-Maps}},
   author = {Dumery, Corentin and Protais, Fran{\c c}ois and Mestrallet, S{\'e}bastien and Bourcier, Christophe and Ledoux, Franck},
   url = {https://doi.org/10.1111/cgf.14649},
   journal = {{Computer Graphics Forum}},
   publisher = {{Wiley}},
   year = {2022},
   month = Aug,
   doi = {10.1111/cgf.14649},
   volume = {41},
   number = {6},
   pages = {467--479},
} 

@article{li2021interactive,
  title={Interactive all-hex meshing via cuboid decomposition},
  author={Li, Lingxiao and Zhang, Paul and Smirnov, Dmitriy and Abulnaga, S Mazdak and Solomon, Justin},
  journal={ACM Transactions on Graphics (TOG)},
  volume={40},
  number={6},
  pages={1--17},
  year={2021},
  publisher={ACM New York, NY, USA}
}
```
</details>
