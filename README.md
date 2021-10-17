# Rice Yield CNN

Rice Yield CNN is ...

## Environment on experiments

### OS

- Ubuntu 18.04.5 LTS

### CPU

- Intel(R) Xeon(R) W-2295 CPU @ 3.00GHz 18 cores

### GPU

- NVIDIA GeForce RTX 3090 x2

### CUDA

- Cuda compilation tools, release 11.3, V11.3.109

### Python

- Python 3.8.8


## Installation

1. Install depentencies.

```bash
pip install -r requirements.txt
```

2. Install Pytorch

Please install pytorch version compatible with your cuda version.

For example, If you use cuda version 11.3,

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```


3. Download pre-trained model from google drive.

```bash
mkdir checkpoints
wget "https://drive.google.com/u/0/uc?export=download&id=1XgTUGK8130gnY9AF3gYv9zhJSJaxhHVp" -O rice_yield_CNN.pth
```

## Estimation

Run

```bash
python estimate.py --checkpoint_path checkpoints/rice_yield_CNN.pth --image_dir example
```

You can find estimated yield on your console.

Below are meanings of options.

- checkpoint_path : Path to the checkpoint file you saved.

- image_dir : path to the directory where images are saved.
