# Dockerfile to build CUDA in Ubuntu

FROM ubuntu:18.04

RUN echo 'Building CUDA environment...'
RUN docker run -it --rm -name cudaEnv -e DISPLAY=$DISPLAY \
  --gpus all nvidia/cuda:10.1-runtime-ubuntu18.04
