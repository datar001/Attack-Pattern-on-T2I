import json
import sys
from PIL import Image
import os
import time
import argparse
from optim_utils import *
from transformers import CLIPModel, CLIPTokenizer, AutoProcessor
from modified_clip import Modified_ClipModel
from synonym import Synonym
import pdb


def write_log(file, text):
    file.write(text + "\n")
    print(text)


def save_top_k_results(outputdir, ori_prompt, prompt_topk):
    save_file = open(os.path.join(outputdir, ori_prompt + '.txt'), "w")
    for sim, mse, prompt, token in prompt_topk:
        write_log(save_file, "sim: {:.3f}, mse: {:.3f}".format(sim, mse))
        write_log(save_file, "prompt: {}".format(prompt))
        write_log(save_file, "token: {}".format(token))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='experiment configuration')

    # load args
    print("Initializing...")
    args = argparse.Namespace()
    args.__dict__.update(read_json(parser.parse_args().config_path))

    # output logger setting
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Modified_ClipModel.from_pretrained(args.clip_model).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model)
    preprocess = AutoProcessor.from_pretrained(args.clip_model)

    # load synonym detection model
    synonym_model = Synonym(word_path="./Word2Vec/", device=device)

    # load all forbidden words
    cur_forbidden_words = open(args.forbidden_words, "r").readlines()[0].lower().split(', ')
    synonym_words = synonym_model.get_synonym(cur_forbidden_words, k=args.synonym_num)
    for word in cur_forbidden_words:
        if len(word.split()) > 1:
            cur_forbidden_words.extend(word.split())
    cur_forbidden_words.extend([word[0] for word in synonym_words])
    print("the forbidden words is {}".format(cur_forbidden_words))

    # load all target goals
    image_dir = args.target_path
    image_names = os.listdir(image_dir)

    # define the output dir for each target goal
    writer_logger = open(os.path.join(output_dir, "logger.txt"), "a+")

    # print the parameter
    write_log(writer_logger, "======  Current Parameter =======")
    for para in args.__dict__:
        write_log(writer_logger, para + ': ' + str(args.__dict__[para]))

    all_learned_prompt = {}
    init_time = time.time()
    # training for each attack goal
    for i, img_name in enumerate(image_names):
        image_id = img_name.split('.')[0]
        if image_id in all_learned_prompt:
            continue
        # load the target image
        orig_images = [Image.open(os.path.join(image_dir, img_name))]
        ori_prompt = ""  #
        print(f"\n======= {i} | {len(image_names)} ====== img id: {image_id}\n")

        object_mask = None

        learned_prompt, adv_text, prompt_topk = optimize_prompt(model, preprocess, tokenizer, args, device,
                                                                ori_prompt=ori_prompt,
                                                                target_images=orig_images,
                                                                forbidden_words=cur_forbidden_words,
                                                                suffix_num=args.token_num,
                                                                only_english_words=False)
        write_log(writer_logger, f"img_id: {image_id}")
        for sim, mse, text, token in prompt_topk:
            write_log(writer_logger, f"sim: {sim}, prompt: {text}")
        if prompt_topk:
            all_learned_prompt[image_id] = [text for sim, mse, text, token in prompt_topk]
        else:
            all_learned_prompt[image_id] = [learned_prompt]
        with open(os.path.join(output_dir, 'AP_FP.json'), 'w') as f:
            json.dump(all_learned_prompt, f, indent=2)



