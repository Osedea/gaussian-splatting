from modal import Image, Stub, Mount, Volume

stub = Stub()

image =  (
    Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu20.04",
        add_python="3.8"
    )
    .apt_install(
        "git",
        "build-essential",
        "gcc-10",
        "g++-10",
        "libglm-dev"
    )
    .workdir("/workspace/")
    .run_commands(
        "pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "tqdm",
        "wheel",
        "plyfile==0.8.1",
    )
    .run_commands(
        "mkdir /workspace/submodules/ && cd /workspace/submodules/ && " \
        "git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git && " \
        "git clone https://gitlab.inria.fr/bkerbl/simple-knn.git "
    )
    .env({
        "CC": "/usr/bin/gcc-10",
        "CXX": "/usr/bin/g++-10"
    })
    .run_commands(
        "python -m pip install submodules/simple-knn/ --no-build-isolation",
        "python -m pip install submodules/diff-gaussian-rasterization/ --no-build-isolation",
        gpu="T4"
    )
)


model_volume = Volume.from_name("model_registry", create_if_missing=True)
data_volume = NetworkFileSystem.from_name("data", create_if_missing=True)
mounts = [
    Mount.from_local_dir(local_dir, remote_path=f"/workspace/{local_dir}")
    for local_dir in [
        "train.py",
        "gaussian_renderer/",
        "scene/",
        "utils/",
    ]
]

@stub.function(
    image=image,
    gpu="T4",
    volumes = {"/workspace/output/": model_volume},
    network_file_systems={"/workspace/data/": data_volume},
    mounts=mounts
)
def f():
    import subprocess

    subprocess.run(["python train.py -s /workspace/data/1/"])

    model_volume.commit()
