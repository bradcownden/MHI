# Dockerfile to build CUDA in Ubuntu

FROM ubuntu:18.04

RUN apt-get -qq update 

# Software to install CUDA
RUN apt-get -qq install wget \ 
      && apt-get -y install curl \
      && apt-get -qq install gnupg \
      && apt-get -qq install software-properties-common \
      && apt-get -qq install openjdk-8-jre
RUN apt-get -y install ubuntu-drivers-common && ubuntu-drivers autoinstall \
      && apt-get -y install linux-headers-$(uname -r)

# Get CUDA Toolkit for Ubuntu (see NVIDIA downloads website)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
RUN echo 'Done! Installing the CUDA Toolkit. This may take several minutes...'
RUN apt-get -qq install cuda

# Clean up apt-get files
RUN echo 'Done! Cleaning up temporary files...'
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /temp/* \ 
      /var/temp/* /cuda-ubuntu1804.pin

# Link Nsight to PATH
RUN PATH=$PATH:/usr/local/cuda-10.2/bin

# Add the Nsight startup script to the home directory
ADD NsightStartup.sh /home

WORKDIR /home





