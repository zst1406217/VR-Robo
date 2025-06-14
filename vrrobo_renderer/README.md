## Installation

In the `./vrrobo_renderer` directory, run the following commands to create a conda environment and install the required packages:
```shell
conda create -n vr-robo-renderer -y python=3.8
conda activate vr-robo-renderer
pip install --upgrade pip

pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

pip install open3d pytorch3d plyfile opencv-python einops e3nn rpyc
pip install submodules/diff-plane-rasterization
pip install submodules/simple-knn
```

Download our captured demo data from [Google Drive](https://drive.google.com/file/d/1qvmFvhSha5FnKgFldW8WFAEO50HhQ7GZ/view?usp=drive_link)
```shell
pip install gdown
gdown 1qvmFvhSha5FnKgFldW8WFAEO50HhQ7GZ && unzip vr-robo-dataset.zip
```
The data folder should like this:
```shell
vrrobo_renderer
├── vr-robo-dataset
│   ├── pcd
│   │   ├── obj1
│   │   │   └── point_cloud.ply
│   │   ├── obj2
│   │   │   └── point_cloud.ply
│   │   │── ...
│   └── transform.json
```

Finally, you can run the render server:
```

## Usage
Start render server:
```shell
python render_server.py
```
