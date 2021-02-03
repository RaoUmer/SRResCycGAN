#!/bin/bash -eux

# Run this script from the repo's root folder:
#
# $ ./docker/build-and-push.sh

# 1. Download pre-trained models from Google Drive

pip install gdown
gdown -O srrescycgan_code_demo/trained_nets_x4/srrescycgan.jpeg-compression.pth 'https://drive.google.com/uc?id=1AJqEm9lfrzkhJf24_ToEOi9iQUpUu2kn'
gdown -O srrescycgan_code_demo/trained_nets_x4/srrescycgan.real-image-corruptions.pth 'https://drive.google.com/uc?id=1NAZjl6UDkcd_BnxfXmwF5QB-uHiJidnA'
gdown -O srrescycgan_code_demo/trained_nets_x4/srrescycgan.sensor-noise.pth 'https://drive.google.com/uc?id=1-N05dWhnA6om16D1VoPASGB9MiTMjdgB'
gdown -O srrescycgan_code_demo/trained_nets_x4/srrescycgan.unknown-compressions.pth 'https://drive.google.com/uc?id=1tPy1LwzRT2LUM2-X3BhGWo4C3Dii1gmV'

# 2. Build Docker images for CPU and GPU

image="us-docker.pkg.dev/replicate/raoumer/srrescycgan"
cpu_tag="$image:cpu"
gpu_tag="$image:gpu"

docker build -f docker/Dockerfile.cpu --tag "$cpu_tag" .
docker build -f docker/Dockerfile.gpu --tag "$gpu_tag" .

# 3. Test the images on sample data

test_input_folder=/tmp/test-srrescycgan/input
mkdir -p $test_input_folder
cp srrescycgan_code_demo/samples/bird.png $test_input_folder/
test_output_folder=/tmp/test-srrescycgan/output

docker run -it \
    -v $test_input_folder:/input \
    -v $test_output_folder/cpu:/output \
    $cpu_tag \
    --input-folder=/input --output-folder=/output

[ -f $test_output_folder/cpu/bird.png ] || exit 1

docker run --gpus all -it \
    -v $test_input_folder:/input \
    -v $test_output_folder/gpu:/output \
    $gpu_tag \
    --input-folder=/input --output-folder=/output

[ -f $test_output_folder/gpu/bird.png ] || exit 1

sudo rm -rf "$test_output_folder"

# 4. Push images to Replicate's Docker registry

docker push "$cpu_tag"
docker push "$gpu_tag"
