import numpy as np
import torch
import cv2
import os
import pandas as pd
import argparse
from glob import glob
from lib.model import RiceYieldCNN
from torch.utils.data.sampler import RandomSampler

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, required=True)
parser.add_argument("--image_dir", type=str, default="example")
parser.add_argument("--csv", action="store_true")

args = parser.parse_args()
checkpoint_path = args.checkpoint_path
image_dir = args.image_dir
csv = args.csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_resolution = (512, 512)
mean = 0.5
std = 0.5

image_path_list = sorted(glob(os.path.join(image_dir, "*")))

if __name__ == "__main__":

    model = RiceYieldCNN()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    results = []
    for i, image_path in enumerate(image_path_list):
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, input_resolution)
        input_img = image.astype(np.float32) / 255.0
        input_img = (input_img - np.array(mean).astype(np.float32)) / \
            np.array(std).astype(np.float32)
        input_img = input_img.transpose(2, 0, 1)
        input_img = torch.Tensor(input_img).unsqueeze(0).to(device)

        # model output
        pred_yield = model(input_img)

        pred_yield = pred_yield[0]
        pred_yield = round(float(pred_yield.squeeze(0).detach().cpu().numpy()), 2)
        print(f"{image_name}: {pred_yield} g/m2, {round(pred_yield / 100,2)} t/ha")

        if csv:
            results.append({
                "id": i,
                "image_name": image_name,
                "gpms": pred_yield,
                "tpha": pred_yield / 100
            })
    if csv:
        pd.DataFrame.from_records(results).to_csv("out.csv", index=False)
