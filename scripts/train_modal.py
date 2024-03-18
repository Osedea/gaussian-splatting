from modal import Image, Mount, NetworkFileSystem, Stub, Volume

stub = Stub()

volume_model = Volume.from_name("model_registry", create_if_missing=True)
volume_data = NetworkFileSystem.from_name("data", create_if_missing=True)

mount_gaussian_splatting = Mount.from_local_dir(
    f"./gaussian_splatting/", remote_path=f"/workspace/gaussian_splatting"
)

image = (
    Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu20.04", add_python="3.8")
    .apt_install("git", "build-essential", "gcc-10", "g++-10", "libglm-dev")
    .workdir("/workspace/")
    .run_commands(
        "pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "tqdm",
        "wheel",
        "plyfile==0.8.1",
    )
    .copy_local_file(local_path="setup.py", remote_path="/workspace/setup.py")
    .run_commands("python -m pip install -e . --no-build-isolation")
    .run_commands(
        "mkdir /workspace/submodules/ && cd /workspace/submodules/ && "
        "git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git && "
        "git clone https://gitlab.inria.fr/bkerbl/simple-knn.git "
    )
    .env({"CC": "/usr/bin/gcc-10", "CXX": "/usr/bin/g++-10"})
    .run_commands(
        "python -m pip install submodules/simple-knn/ --no-build-isolation",
        "python -m pip install submodules/diff-gaussian-rasterization/ --no-build-isolation",
        gpu="T4",
    )
)


@stub.function(
    image=image,
    gpu="T4",
    volumes={"/workspace/output/": volume_model},
    network_file_systems={"/workspace/data/": volume_data},
    mounts=[mount_gaussian_splatting],
    timeout=10800,
)
def f():
    from gaussian_splatting.training import Trainer

    trainer = Trainer(source_path="/workspace/data/phil_open/5")
    trainer.run()

    volume_model.commit()
