import torch
import argparse
import torch.nn as nn
from transformers import CLIPModel, AutoTokenizer, AutoProcessor
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from classification import ClassificationModel
import os, json
import time
from typing import Any, Mapping
import numpy as np
import random
import pdb

def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)

def save_image(image, path):
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)
    image.save(path)

def metric(probs, gt, return_false=False):
    bs = probs.size(0)
    max_v, max_index = torch.max(probs, dim=-1)
    acc = (max_index == gt).sum()
    if return_false:
        # pdb.set_trace()
        false_index = torch.where(max_index != gt)[0]
        return acc, false_index
    return acc

def write_log(file, text):
    file.write(text + "\n")
    print(text)

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--adversarial_prompt_dir', type=str, required=True, help='the path of adversarial prompts')
    parser.add_argument('--output_image_dir', type=str, required=True, help='the path of generated images')
    parser.add_argument('--attack_model', type=str, default="SD2.1", choices=["SD1.4", "SD1.5", "SD2.1", "SDXL"], 
                        help='Choose a text-to-image model to craft attacks')
    args = argparse.Namespace()

    # define the image output dir
    print("generated path:{}".format(args.output_image_dir))
    if os.path.exists(args.output_image_dir):
        replace_type = input("The image output path has existed, replace all? (yes/no) ")
        if replace_type == "no":
            exit()
        elif replace_type == "yes":
            pass
        else:
            raise ValueError("Answer must be yes or no")
    os.makedirs(args.output_image_dir, exist_ok=True)

    # load labels of clean prompts
    label_path = "./mini_100.txt"
    with open(label_path, 'r') as f:
        label_infos = f.readlines()
    label_infos = [label.lower().strip() for label in label_infos]

    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.attack_model == "SD1.4":
        model_id = "CompVis/stable-diffusion-v1-4"
    elif args.attack_model == "SD1.5":
        model_id = "runwayml/stable-diffusion-v1-5"
    elif args.attack_model == "SD2.1":
        model_id = "stabilityai/stable-diffusion-2-1-base"
    else:
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        revision="fp16",
    )
    pipe = pipe.to(device)
    image_length = 512

    # load classification model
    classify_model = ClassificationModel(model_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                                         label_txt='./mini_100.txt', device=device, mode='object')
    total_5_acc, total_10_acc, total_acc = 0, 0, 0
    gen_num = 10
    batch_size = 5
    batch = int(np.ceil(gen_num / batch_size))
    attack_goal_num = 0
    for attack_target in os.listdir(args.output_dir):
        object_result_path = os.path.join(args.output_dir, attack_target)
        if not os.path.isdir(object_result_path):
            continue
        attack_goal_num += 1
        attack_path = os.path.join(object_result_path, "results.txt")
        with open(attack_path, 'r') as f:
            attack_infos = f.readlines()
        # image output dir
        cur_output_img_dir = os.path.join(args.output_image_dir, attack_target)
        if os.path.exists(cur_output_img_dir):
            replace_type = input("The output acc path has existed!!! replace all? (yes/no) ")
            if replace_type == "no":
                exit()
            elif replace_type == "yes":
                pass
            else:
                raise ValueError("Answer must be yes or no")
        os.makedirs(cur_output_img_dir, exist_ok=True)
        output_file = open(os.path.join(cur_output_img_dir, "results.txt"), "w")

        # add the target attack to the labels of the classification
        if attack_target in classify_model.labels:
            attack_target_id = classify_model.labels.index(attack_target.lower())
        else:
            classify_model.add_label_param([attack_target.lower()])
            attack_target_id = classify_model.labels.index(attack_target.lower())

        # generate images
        each_5_acc, each_10_acc, each_acc = 0, 0, 0
        init_time = time.time()
        for i in range(1, len(attack_infos)):
            set_random_seed(666)
            label = label_infos[i-1].strip()  # the label of the clean prompt
            prompt = attack_infos[i].strip()
            write_log(output_file, "Generate {}^th adv prompt: {}, ori label:{}, attack target: {}".format(
                i, prompt, label, attack_target))
            assert label.replace("-", " - ") in prompt, "The adversarial prompt don't contain the original object," \
                                                             f"current object: {label}, current prompt: {prompt}"
            cur_5_acc, cur_10_acc, cur_avg_acc = 0, 0, 0
            start_time = time.time()
            for j in range(batch):
                num_images = min(gen_num, (j+1)*batch_size) - j*batch_size
                guidance_scale = 9
                num_inference_steps = 25
                images = pipe(
                    prompt,
                    num_images_per_prompt=num_images,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    height=image_length,
                    width=image_length,
                    ).images


                probs = classify_model.forward(images)
                acc_num = metric(probs, attack_target_id)
                cur_avg_acc += acc_num
                if j == 0 and acc_num > 0:
                    cur_5_acc = 1
                if acc_num > 0:
                    cur_10_acc = 1
                for img_num in range(num_images):
                    sign = probs[img_num].argmax(0) == attack_target_id
                    dir_name = prompt.replace(" ", "_")
                    save_image(images[img_num],
                               os.path.join(cur_output_img_dir,
                                            f"{label}/{dir_name}/{sign}_{img_num+j*batch_size}.png"))
            end_time = time.time()
            each_5_acc += cur_5_acc
            each_10_acc += cur_10_acc
            each_acc += cur_avg_acc
            write_log(output_file, f"{label} ASR@5 is: {cur_5_acc}, ASR@10 is {cur_10_acc}")
            write_log(output_file, "Spent time: {:.3f}s".format(end_time - start_time))
            write_log(output_file, "The ASR@1 is {:.3f}".format(cur_avg_acc))
        write_log(output_file, f"\nEnd the testing stage of an attack goal: {attack_target}\n")
        write_log(output_file, "ASR@5 is {:.3f}%".format(each_5_acc * 100 / (len(attack_infos) - 1)))
        write_log(output_file, "ASR@10 is {:.3f}%".format(each_10_acc * 100 / (len(attack_infos) - 1)))
        write_log(output_file, "ASR@1 is {:.3f}%".format(each_acc * 100 / gen_num / (len(attack_infos) - 1)))
        finish_time = time.time()
        all_time = finish_time - init_time
        write_log(output_file,
                  "spent time is {}h{}m".format(all_time // 3600, (all_time % 3600) // 60))
        total_5_acc += each_5_acc * 100 / (len(attack_infos) - 1)
        total_10_acc += each_10_acc * 100 / (len(attack_infos) - 1)
        total_acc += each_acc * 100 / gen_num / (len(attack_infos) - 1)
    total_result_path = os.path.join(args.output_image_dir, "results.txt")
    output_file = open(total_result_path, "w")
    write_log(output_file, f"\nEnd the all testing stage\n")
    write_log(output_file, "Final ASR@5 is {:.3f}%".format(total_5_acc / attack_goal_num))
    write_log(output_file, "Final ASR@10 is {:.3f}%".format(total_10_acc / attack_goal_num))
    write_log(output_file, "Final ASR@1 is {:.3f}%".format(total_acc / attack_goal_num))