# Rice Yield CNN

Rice Yield CNN is ...

## Environment on experiments

### OS

- Ubuntu 18.04.5 LTS

### CPU

- Intel(R) Xeon(R) W-2295 CPU @ 3.00GHz 18 cores

### GPU

- NVIDIA GeForce RTX 3090 x2

### Python

- Python 3.8.8


## Installation

1. Install depentencies.

```bash
pip install -r requirements.txt
```

2. Download pre-trained model from google drive.

```bash
mkdir models
wget "https://drive.google.com/u/0/uc?export=download&id=1XgTUGK8130gnY9AF3gYv9zhJSJaxhHVp" -O rice_yield_CNN.pth
```

3. Start yield estimation with example images.

```bash
python estimate.py --images data/example/
```

