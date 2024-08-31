import pdb
import torch
import torch.nn as nn
import numpy as np
import clip
from PIL import Image
import argparse
from transformers import CLIPModel, AutoTokenizer, AutoProcessor
import os, random
import tqdm
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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

class H14_Malicious_Prompt_Detector(nn.Module):
    def __init__(self, device='cpu', input_size=1024):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.Linear(1024, 2048),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(2048, 1024),
            # nn.ReLU(),
            # nn.Dropout(0.2),
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

    def load_model(self, model_path='./h14_nsfw.pth'):
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)


def get_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        data = json.load(f)
    prompt_data = []
    for id, info in data.items():
        if type(info['prompt']) == list:
            prompt_data.extend(info['prompt'])
        else:
            prompt_data.append(info['prompt'])
    return prompt_data


def get_data():
    ## malicious data
    train_mal_prompt_path = "/home/zcy/attack/defense_dataset/mal_all_data/train_data.json"
    train_mal_prompt = get_prompt(train_mal_prompt_path)
    train_mal_label = [1] * len(train_mal_prompt)
    val_mal_prompt_path = "/home/zcy/attack/defense_dataset/mal_all_data/val_data.json"
    val_mal_prompt = get_prompt(val_mal_prompt_path)
    val_mal_label = [1] * len(val_mal_prompt)

    ## clean data
    train_clean_prompt_path = "/home/zcy/attack/defense_dataset/clean_data/filter-8/train_data.json"
    train_clean_prompt = get_prompt(train_clean_prompt_path)
    train_clean_label = [0] * len(train_clean_prompt)
    val_clean_prompt_path = "/home/zcy/attack/defense_dataset/clean_data/filter-8/val_data.json"
    val_clean_prompt = get_prompt(val_clean_prompt_path)
    val_clean_label = [0] * len(val_clean_prompt)

    train_data = []
    train_data.extend(train_mal_prompt)
    train_data.extend(train_clean_prompt)
    train_label = []
    train_label.extend(train_mal_label)
    train_label.extend(train_clean_label)

    val_data = []
    val_data.extend(val_mal_prompt)
    val_data.extend(val_clean_prompt)
    val_label = []
    val_label.extend(val_mal_label)
    val_label.extend(val_clean_label)

    return train_data, train_label, val_data, val_label

def get_adv_data():
    with open("/home/zcy/attack/defense_dataset/mal_adv_prompt/train/all_word_substitution.json") as f:
        adv_pattern_1 = json.load(f)
    new_adv_pattern_1 = {}
    for id, data in adv_pattern_1.items():
        new_adv_pattern_1[data["prompt"]] = list(data["adv_prompt"].values())
    with open("/home/zcy/attack/defense_dataset/mal_adv_prompt/train/all_random_words.json") as f:
        adv_pattern_2 = json.load(f)
    with open("/home/zcy/attack/defense_dataset/mal_all_data/train_data.json", "r") as f:
        mal_data = json.load(f)
    new_adv_pattern_2 = {}
    for id, data in mal_data.items():
        if data["image_url"] == "None":
            image_id = id
        else:
            image_id = data["image_url"].split('/')[-1]
        new_adv_pattern_2[data["prompt"]] = adv_pattern_2[image_id]
        assert id in adv_pattern_1, print(f"id: {id}, not in malicious data")
        assert image_id in adv_pattern_2, print(f"image id: {image_id}, not in malicious data")
    return new_adv_pattern_1, new_adv_pattern_2

def get_adv_train_data(cur_batch_data, adv_prompt_1, adv_prompt_2, thres1, thres2):
    new_batch_data = []
    for data in cur_batch_data:
        if data in adv_prompt_1:  # if data is a malicious prompt
            if torch.rand(1).item() > thres1:  # replace adv prompt with mal prompt
                if torch.rand(1).item() > thres2:  # replace adv_prompt_1
                    if len(adv_prompt_1[data]) == 0:
                        new_batch_data.append(data)
                    else:
                        random_id = np.random.randint(0, len(adv_prompt_1[data]), size=[1])
                        new_batch_data.append(adv_prompt_1[data][random_id[0]])
                else:
                    random_id = np.random.randint(0, len(adv_prompt_2[data]), size=[1])
                    new_batch_data.append(adv_prompt_2[data][random_id[0]])
            else:
                new_batch_data.append(data)
        else:  # if data is a clean prompt, do not conduct the replacement
            new_batch_data.append(data)
    return new_batch_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='the number of a batch prompt')
    parser.add_argument('--gt', type=str, default='unsafe',
                        help='the image type')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data
    train_data, train_label, val_data, val_label = get_data()
    # train_label = torch.tensor(train_label).view(-1, 1).to(device)
    val_label = torch.tensor(val_label).view(-1, 1).to(device)

    # load model
    clip_model = CLIP_Model(device)
    classify_model = H14_Malicious_Prompt_Detector(device)

    # load optimizer
    optimizer = torch.optim.AdamW(classify_model.parameters(), lr=args.lr)

    # load loss function
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits, suitable for binary classification

    # training parameters
    train_num = len(train_data)
    batch_size = args.batch_size
    batch_num = train_num // batch_size
    if train_num % batch_size != 0:
        batch_num += 1
    num_epochs = args.num_epochs
    save_model_dir = "./checkpoint/adv_train_sexy%vio_08_04_1e3_epoch25"
    os.makedirs(save_model_dir, exist_ok=True)
    best_acc = 0

    # Training loop
    for epoch in range(num_epochs):

        # shuffle training data
        shuffle_train_data, shuffle_train_label = [], []
        shuffle_id = np.random.choice(np.arange(len(train_data)), len(train_data), replace=False)
        for id in shuffle_id:
            shuffle_train_data.append(train_data[id])
            shuffle_train_label.append(train_label[id])
        shuffle_train_label = torch.tensor(shuffle_train_label).view(-1, 1).to(device)

        # load adv_data
        adv_prompt_1, adv_prompt_2 = get_adv_data()

        # randomly replace malicious prompt with adv prompt
        shuffle_train_data_fuse = get_adv_train_data(shuffle_train_data, adv_prompt_1, adv_prompt_2, 0.5, 0.6)

        classify_model.train()
        epoch_loss = 0.0

        for i in tqdm.tqdm(range(batch_num), desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
            optimizer.zero_grad()
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, train_num)
            cur_batch_data = shuffle_train_data_fuse[start_idx:end_idx]
            # cur_batch_data = get_adv_train_data(cur_batch_data, adv_prompt_1, adv_prompt_2, 0.5, 0.6)
            cur_batch_label = shuffle_train_label[start_idx:end_idx].float()

            # forward
            cur_batch_feature = clip_model(cur_batch_data)
            preds = classify_model(cur_batch_feature)
            # Compute loss
            loss = criterion(preds, cur_batch_label)  # Assuming train_label is binary (0 or 1)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(cur_batch_data)

        epoch_loss /= train_num
        print(f'Training Loss: {epoch_loss:.4f}')

        # Validation (assuming val_data and val_label are defined similarly)
        classify_model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            val_preds = classify_model(clip_model(val_data))
            val_loss = criterion(val_preds, val_label.float())
            val_preds = val_preds.cpu().numpy()  # Convert logits to probabilities
            val_preds = (val_preds > 0.5).astype(int)  # Convert probabilities to binary predictions
            accuracy = (val_preds == val_label.cpu().numpy()).mean()

            print(f'Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {accuracy:.4f}')

            if accuracy > best_acc:
                best_acc = accuracy
                model_path = os.path.join(save_model_dir, "best.pth")
                torch.save(classify_model.state_dict(), model_path)

