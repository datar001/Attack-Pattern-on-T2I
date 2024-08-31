import torch
import torch.nn as nn
import argparse
from transformers import CLIPModel, AutoTokenizer, AutoProcessor
import os
import tqdm
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import random
import numpy as np
import pandas as pd
import pdb

class CLIP_Model(nn.Module):
    def __init__(self, device='cpu'):
        super(CLIP_Model, self).__init__()
        self.model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model.to(device)
        self.device = device

    def forward(self, prompts):
        with torch.no_grad():
            inputs = self.processor(text=prompts, padding=True, return_tensors="pt")

            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            text_embeds = self.model.get_text_features(**inputs)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        return text_embeds

class Prompt_Detector(nn.Module):
    def __init__(self, device='cpu', input_size=1024):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.Linear(128, 16),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.layers.to(device)

    def forward(self, x):
        return self.layers(x)

    def load_model(self, model_path='./checkpoint/best.pth'):
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="", required=True, help='the input prompt')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/best.pth',
                        help='the model checkpoint')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load model
    clip_model = CLIP_Model(device)
    classify_model = Prompt_Detector(device)
    classify_model.load_model(args.checkpoint)
    classify_model.eval()  # Set model to evaluation mode


    with torch.no_grad():
        pred = classify_model(clip_model(args.prompt))[0]

    print(f"the probability of input prompt having maliciousness is {pred.cpu().item()}")
