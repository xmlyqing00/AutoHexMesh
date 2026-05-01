# enable GUI program
xhost +local:root

sudo docker run \
--runtime=nvidia \
--gpus all \
-it \
--env="DISPLAY=$DISPLAY" \
--env="NVIDIA_DRIVER_CAPABILITIES=all" \
--env="NVIDIA_VISIBLE_DEVICES=0" \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
-v /usr/share/vulkan:/usr/share/vulkan:ro \
-v $(pwd)/lib:/space/lib \
-v $(pwd)/evocube:/space/evocube \
-v $(pwd)/interactive-hex-meshing:/space/interactive-hex-meshing \
-v $(pwd)/compile.sh:/space/compile.sh \
-v $(pwd)/data:/space/data \
-v $(pwd)/output:/space/output \
docker-hexmesh
