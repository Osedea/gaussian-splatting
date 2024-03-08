from modal import Image, Stub, Mount, Volume

stub = Stub()

image =  (
    Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu20.04",
        add_python="3.8"
    )
    .apt_install("git", "build-essential")
    .workdir("/workspace/")
    .run_commands(
        "pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "tqdm",
        "plyfile==0.8.1",
    )
    .run_commands(
        "mkdir /workspace/submodules/ && cd /workspace/submodules/ && " \
        "git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git && " \
        "git clone https://gitlab.inria.fr/bkerbl/simple-knn.git "
    )
    .run_commands(
        "pip install submodules/simple-knn/",
        "pip install submodules/diff-gaussian-rasterization",
        gpu="T4"
    )
)


#volume = Volume.from_name("model_registry_volume", create_if_missing=True)
#mounts = [
#    Mount.from_local_dir(local_dir, remote_path=f"/workspace/{local_dir}")
#    for local_dir in [
#        "data/",
#        "train.py",
#        "gaussian_renderer/",
#        "scene/",
#        "utils/",
#        "output/"
#    ]
#]
@stub.function(
    image=image,
    gpu="T4",
    #volumes = {"/workspace/output/": volume}
    #modal_mounts=mounts
)
def f():
    import torch

    print(torch.cuda.is_available())

 #   subprocess.run(["python train.py -s /workspace/data/"])

 #   vol.commit()
