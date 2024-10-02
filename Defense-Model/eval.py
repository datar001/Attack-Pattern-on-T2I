import torch
import torch.nn as nn
import clip
from PIL import Image
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
from matplotlib import font_manager
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
            inputs = self.processor(text=prompts, padding=True, return_tensors="pt", truncation = True)

            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            text_embeds = self.model.get_text_features(**inputs)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        return text_embeds

class Prompt_Classifier(nn.Module):
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
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.layers.to(device)

    def forward(self, x):
        return self.layers(x)

    def load_model(self, model_path='./h14_nsfw.pth'):
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)

def plot_auroc_for_compare(y_tests, y_scores, names):
    for y_test, y_score, name in zip(y_tests, y_scores, names):
        auc_mean, auc_std, auc_score = get_auroc_data(y_test, y_score, name)
        get_fpr90(y_test, y_score, name, thres=0.9)

def get_fpr90(y_test, y_score, mode, thres=0.90):
    global df
    fprs = []
    for i in range(len(y_test)):
        fpr, tpr, thresholds = roc_curve(y_test[i], y_score[i])
        target_tpr = thres
        closest_tpr_index = np.argmin(np.abs(tpr - target_tpr))  # 找到最接近95% TPR的索引
        fpr_at_tpr90 = fpr[closest_tpr_index]  # 对应的FPR
        fprs.append(fpr_at_tpr90)
    fpr_mean = np.mean(fprs)
    fpr_std = np.std(fprs)
    print(f"{mode}, mean fpr@tpr{int(thres*100)}: {fpr_mean}, std fpr@tpr{int(thres*100)}: {fpr_std}")
    df.loc[f'FPR@TPR{int(thres*100)}', mode] = f'{100 * fpr_mean:.2f}±{100 * fpr_std:.2f}'

def get_auroc_data(y_test, y_score, mode):
    global df
    auc_scores = [roc_auc_score(y_test[i], y_score[i]) for i in range(len(y_test))]
    auc_mean = np.mean(auc_scores)
    auc_std = np.std(auc_scores)
    print(f"{mode}, mean auc: {auc_mean}, std auc: {auc_std}")
    df.loc['AUROC', mode] = f'{100*auc_mean:.2f}±{100*auc_std:.2f}'

    return auc_mean, auc_std, auc_scores

def get_mal_json_data(data_path):
    with open(data_path, 'r') as f:
        mal_data = json.load(f)
    test_mal_data = []

    for id, info in mal_data.items():
        if type(info['adv_prompt']) == list:
            test_mal_data.extend(info['adv_prompt'])
        else:
            test_mal_data.append(info['adv_prompt'])
    return test_mal_data

def get_csv_data(data_path):
    try:
        infos = pd.read_csv(data_path, encoding="utf-8")
    except:
        infos = pd.read_csv(data_path, encoding="ISO-8859-1")
    prompts = infos["prompt"].to_list()
    return prompts

def sample_data(mal_data, clean_data, sample_num):
    mal_num, clean_num = len(mal_data), len(clean_data)
    new_mal_data, new_clean_data = [], []
    if sample_num > 0:
        for _ in range(sample_num):
            if mal_num > clean_num:
                clean_data_copy = []
                new_clean_num = clean_num
                clean_data_copy.extend(clean_data)
                while new_clean_num != mal_num:
                    extend_num = mal_num - new_clean_num
                    if extend_num > clean_num:
                        clean_data_copy.extend(random.sample(clean_data, clean_num))
                        new_clean_num += clean_num
                    else:
                        clean_data_copy.extend(random.sample(clean_data, extend_num))
                        new_clean_num += extend_num
                new_clean_data.append(clean_data_copy)
                new_mal_data.append(mal_data)
                assert len(clean_data_copy) == len(mal_data)
            else:
                clean_data_copy = random.sample(clean_data, mal_num)
                new_clean_data.append(clean_data_copy)
                new_mal_data.append(mal_data)
                assert len(clean_data_copy) == len(mal_data)
    else:
        new_mal_data, new_clean_data = [mal_data], [clean_data]
    return new_mal_data, new_clean_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_sample_num', type=int, default=5,
                        help='sampling clean prompts to match the number of malicious prompts')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='the number of a batch image')
    parser.add_argument('--checkpoint', type=str, default='./',
                        help='evaluation metric')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load model
    clip_model = CLIP_Model(device)
    classify_model = Prompt_Classifier(device)
    classify_model.load_model(args.checkpoint)
    classify_model.eval()  # Set model to evaluation mode

    FP_path = "./data/Feature_Pattern.json"
    TP_path = "./data/Text_Pattern.json"
    Sneak_path = "./data/SneakyPrompt.csv"
    UnlearnDiff_path = "./data/UnlearnDiff.csv"
    MMA_path = "./data/MMA.csv"
    P4D_path = "./data/P4D.csv"

    # mal_data = get_mal_json_data(mal_path)
    FP_data = get_mal_json_data(FP_path)
    TP_data = get_mal_json_data(TP_path)
    Sneak_data = get_csv_data(Sneak_path)
    UnlearnDiff_data = get_csv_data(UnlearnDiff_path)
    MMA_data = get_csv_data(MMA_path)
    P4D_data = get_csv_data(P4D_path)

    clean_prompt_path = "./data/Clean_Prompt.json"
    with open(clean_prompt_path, "r") as f:
        test_clean_data = json.load(f)
    clean_data = []
    for id, info in test_clean_data.items():
        if type(info['prompt']) == list:
            clean_data.extend(info['prompt'])
        else:
            clean_data.append(info['prompt'])

    # mal_clean_negative, mal_clean_positive = sample_data(mal_data, clean_data, args.clean_sample_num)
    FP_clean_negative, FP_clean_positive = sample_data(FP_data, clean_data, args.clean_sample_num)
    TP_clean_negative, TP_clean_positive = sample_data(TP_data, clean_data, args.clean_sample_num)
    Sneak_clean_negative, Sneak_clean_positive = sample_data(Sneak_data, clean_data, args.clean_sample_num)
    UnlearnDiff_clean_negative, UnlearnDiff_clean_positive = sample_data(UnlearnDiff_data, clean_data, args.clean_sample_num)
    MMA_clean_negative, MMA_clean_positive = sample_data(MMA_data, clean_data, args.clean_sample_num)
    P4D_clean_negative, P4D_clean_positive = sample_data(P4D_data, clean_data, args.clean_sample_num)

    all_test_data = [
                 (TP_clean_negative, TP_clean_positive),
                (FP_clean_negative, FP_clean_positive),
                (Sneak_clean_negative, Sneak_clean_positive),
                (UnlearnDiff_clean_negative, UnlearnDiff_clean_positive),
                (MMA_clean_negative, MMA_clean_positive),
                (P4D_clean_negative, P4D_clean_positive),
    ]

    modes = ['TextPattern', 'FeatruePattern', 'SneakPrompt', 'UnlearnDiff', 'MMA', 'P4D']
    df = pd.DataFrame(index=['ACC', 'FPR@TPR90', 'AUROC'], columns=['TextPattern', 'FeatruePattern', 'SneakPrompt', 'UnlearnDiff', 'MMA', 'P4D'])
    all_labels, all_preds = [], []
    for mode_id, (negative_datas, positive_datas) in enumerate(all_test_data):
        test_labels, test_preds = [], []
        final_acc = []
        for (negative_data, positive_data) in zip(negative_datas, positive_datas):
            test_data = []
            test_data.extend(negative_data)
            test_data.extend(positive_data)
            test_label = []
            test_label.extend([1]*len(negative_data))
            test_label.extend([0]*len(positive_data))
            prompt_num = len(test_data)
            batch_size = args.batch_size
            batch_num = prompt_num // batch_size
            if prompt_num % batch_size != 0:
                batch_num += 1
            test_pred = []
            for i in range(batch_num):
                cur_batch_paths = test_data[i*batch_size: min((i+1)*batch_size, prompt_num)]
                with torch.no_grad():
                    preds = classify_model(clip_model(cur_batch_paths))
                    for pred in preds:
                        test_pred.append(pred.detach().cpu())

            test_label = torch.tensor(test_label).view(-1, 1)
            test_pred = torch.tensor(test_pred).view(-1, 1)
            avg_acc = ((test_pred > 0.5).float() == test_label).sum().item() / prompt_num
            final_acc.append(avg_acc)
            test_labels.append(test_label)
            test_preds.append(test_pred)
        final_avg_acc = np.mean(np.array(final_acc))
        final_std_acc = np.std(np.array(final_acc))
        print(f"{modes[mode_id]}, mean acc: {final_avg_acc}, std acc: {final_std_acc}")
        df.loc['ACC', modes[mode_id]] = f'{100*final_avg_acc:.2f}±{100*final_std_acc:.2f}'
        all_labels.append(test_labels)
        all_preds.append(test_preds)

    plot_auroc_for_compare(all_labels, all_preds, modes)
    df.to_csv("./result.csv", encoding="utf-8-sig")