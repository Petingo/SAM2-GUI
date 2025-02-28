# SAM2 GUI
A GUI for easy interaction with Meta's SAM2 video segmentation model

![](screenshots/demo.gif)

## Preparation
1. Setup the environment and install [SAM2](https://ai.meta.com/sam2/). If you already have it installed, skip this step.
```bash
# create conda environment
conda create -n sam2 python=3.10
conda activate sam2

# clone the SAM2 repo & install the required package
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .

# download the checkpoints
cd checkpoints
bash download_ckpts.sh
```

2. Clone this project and install the required package
```bash
git clone https://github.com/Petingo/SAM2-GUI.git && cd SAM2-GUI
pip install -r requirements.txt
```

## Usage
1. Run `sam2_app.py`. You can assign the model and checkpoint to use.
```
python sam2_app.py \
    --checkpoint_dir ../sam2/checkpoints/sam2.1_hiera_large.pt \
    --model_cfg ../sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
    --port 8890
```

2. Open your browser and goes to http://127.0.0.1:8890/

## Troubleshooting 
The working directory is a bit messed up right now. If you encounter problems like checkpoint not found or model config not found, please arrange the directory structure as following (make sure that all files exist!) and run `sam2_gui.py` using the default setting:

```
.
├── sam2/
│   ├── checkpoints/
│   │   ├── sam2.1_hiera_base_plus.pt
│   │   ├── sam2.1_hiera_large.pt
│   │   ├── sam2.1_hiera_small.pt
│   │   └── sam2.1_hiera_tiny.pt
│   └── sam2/
│       └── configs/
│           └── sam2.1/
│               ├── sam2.1_hiera_b+.yaml
│               ├── sam2.1_hiera_l.yaml
│               ├── sam2.1_hiera_s.yaml
│               └── sam2.1_hiera_t.yaml
└── SAM2-GUI/
    └── sam2_gui.py
```

## Acknoledge
- This project is modified from [SAM2-GUI by YunxuanMao](https://github.com/YunxuanMao/SAM2-GUI), but fixes some bugs and improved the UI.