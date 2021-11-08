# Rice Yield CNN

Rice Yield CNN is a model to estimate the rice yield based on RGB image of rice canopy at harvest. The model is developed based on more than 22,000 images and yield database collected across 7 countries.  
This project is the implementation of the paper "[Deep learning-based estimation of rice yield using RGB image](https://www.researchsquare.com/article/rs-1026695/v1)".

## Performance 

The model explained approximately 70% of variation in observed rice yield using the test dataset, and 50% of variation using the independent prediction dataset. The model is also able to forecast the rice yield approximately 10-20 days before harvest and is practically robust to the brightness, contrast or angle of the RGB image .


## Conditions on estimation

RGB images that were captured vertically downwards over the rice canopy from a distance of 0.8 to 0.9 m using a digital camera should be input. 

![example](https://github.com/r1wtn/rice_yield_CNN/blob/develop/example/1.jpg)

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
python estimate.py --checkpoint_path checkpoints/rice_yield_CNN.pth --image_dir example --csv
```

You can find estimated yield on your console.

Below are meanings of options.

- checkpoint_path : Path to the checkpoint file you saved.

- image_dir : path to the directory where images are saved.

- csv: If you set this, csv of results will be generated.
