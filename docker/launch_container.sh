docker run --ulimit nofile=65535:65535 --shm-size=8g --gpus all -it --rm --volume /:/host --workdir /host$PWD mvsgaussian
