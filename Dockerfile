FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

# Gaussian splt dependencies
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    wget \
    vim \
    imagemagick

# Install conda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
	/bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /workspace

# Install requirements
COPY setup.py setup.py
COPY environment.yml environment.yml
COPY gaussian_splatting/ gaussian_splatting/ 
COPY scripts/ scripts/ 
COPY submodules/ submodules/ 
RUN conda init
RUN conda env create --file environment.yml

# COLMAP dependencies
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \ 
    cmake \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libflann-dev \
    libsqlite3-dev \
    ninja-build \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libceres-dev

#Build COLMAP
RUN cd .. && \
    git clone https://github.com/colmap/colmap.git && \
    cd colmap && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja && \
    ninja && \
    ninja install

