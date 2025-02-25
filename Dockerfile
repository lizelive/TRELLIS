# syntax=docker/dockerfile:1.13
FROM ghcr.io/mamba-org/micromamba:2.0.5 as base
USER root
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
ENV TORCH_CUDA_ARCH_LIST=8.0;8.9

# install wget and git using apt
# use cache to speed up the build
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt  apt-get update && apt-get install -y wget git


RUN --mount=type=cache,target=/opt/conda/pkgs --mount=type=cache,target=/root/.cache/pip  CONDA_OVERRIDE_CUDA="12.6" micromamba install -y -n base -f /tmp/env.yaml

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics


# git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git