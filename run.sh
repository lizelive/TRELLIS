#!/bin/sh
podman run --rm -it --device nvidia.com/gpu=all --user 0:0 -p 7860:7860 -v ~/.cache/huggingface:/root/.cache/huggingface -v torch-cache:/root/.cache/torch -v ./:/root/TRELLIS -w /root/TRELLIS --platform=linux/amd64 localhost/trellis:latest bash

# 

# podman run --rm -it --device nvidia.com/gpu=all --user 0:0 -p 7860:7860 -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.cache/torch:/root/.cache/torch -v ./:/home/user/TRELLIS  --platform=linux/amd64 registry.hf.space/jeffreyxiang-trellis:latest bash