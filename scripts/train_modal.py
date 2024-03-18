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


class Dataset:
    def __init__(
        self,
    ):
        self.sh_degree = 3
        self.source_path = "/workspace/data/phil_open/5"
        self.model_path = ""
        self.images = "images"
        self.resolution = -1
        self.eval = False


class Optimization:
    def __init__(self):
        self.iterations = 30000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densify_grad_threshold = 0.0002


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

    trainer = Trainer()
    trainer.run(
        dataset=Dataset(),
        opt=Optimization(),
    )

    volume_model.commit()
