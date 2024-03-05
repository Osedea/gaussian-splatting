FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

# gcc compiler
RUN apt update && \
    apt-get install -y build-essential wget vim

# Install conda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
	/bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

COPY environment.yml environment.yml
COPY submodules/ submodules/ 
RUN conda init
RUN conda env create --file environment.yml

WORKDIR /workspace
