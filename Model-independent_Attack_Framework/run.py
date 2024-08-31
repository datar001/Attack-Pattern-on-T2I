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

def write_log(file, text, print_console=True):
    file.write(text + "\n")
    if print_console:
        print(text)

def save_top_k_results(outputdir, ori_prompt, prompt_topk):
    save_file = open(os.path.join(outputdir, ori_prompt + '.txt'), "w")
    for k, (sim, mse, prompt, token) in enumerate(prompt_topk):
        if k > len(prompt_topk) - 10:
            print_console = True
        else:
            print_console = False
        write_log(save_file, "sim: {:.3f}, mse: {:.3f}".format(sim, mse), print_console)
        write_log(save_file, "prompt: {}".format(prompt), print_console)
        write_log(save_file, "token: {}".format(token), print_console)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='experiment configuration')

    # load args
    print("Initializing...")
    args = argparse.Namespace()
    args.__dict__.update(read_json(parser.parse_args().config_path))

    # output logger setting
    output_dir = args.output_dir
    if os.path.exists(output_dir):
        replace_type = input("The output path has existed, replace all? (yes/no) ")
        if replace_type == "no":
            exit()
        elif replace_type == "yes":
            pass
        else:
            raise ValueError("Answer must be yes or no")
    os.makedirs(output_dir, exist_ok=True)

    # load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Modified_ClipModel.from_pretrained(args.clip_model).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model)
    preprocess = AutoProcessor.from_pretrained(args.clip_model)

    # load synonym detection model
    synonym_model = Synonym(word_path="./Word2Vec/", device=device)

    # load all attack goals
    with open(args.target_path, 'r') as f:
        attack_targets = f.readlines()
    attack_targets = [goal.strip() for goal in attack_targets]

    # load all forbidden words~(sensitive words)
    with open(args.forbidden_words, "r") as f:
        forbidden_words = f.readlines()
    forbidden_words = [words.strip().split(',') for words in forbidden_words]

    assert len(attack_targets) == len(forbidden_words), "The number of target goals must equal to the number " \
                                                      f"of forbidden words, but get {len(attack_targets)} target goals " \
                                                      f"and {len(forbidden_words)} forbidden words"

    # load imagenet-mini label -- the category of clean prompts
    object_path = r"./mini_100.txt"
    with open(object_path, "r") as f:
        objects = f.readlines()
    objects = [obj.strip() for obj in objects]
        
    # load clean prompts
    with open(args.prompt_path, 'r') as f:
        clean_prompts = f.readlines()

    # generating adversarial prompt for each attack target
    for cur_forbidden_words, goal_path in zip(forbidden_words, attack_targets):
        attack_tgt = goal_path.split('/')[-1]
        print('\n\tStart to train a new attack target: {}\n'.format(attack_tgt))

        # load the targeted image
        orig_images = [Image.open(os.path.join(goal_path, image_name)) for image_name in os.listdir(goal_path)]
        cur_output_dir = os.path.join(output_dir, attack_tgt)
        if os.path.exists(cur_output_dir):
            replace_type = input("The adv prompt output path has existed, replace all? (yes/no) ")
            if replace_type == "no":
                exit()
            elif replace_type == "yes":
                pass
            else:
                raise ValueError("Answer must be yes or no")
        os.makedirs(cur_output_dir, exist_ok=True)
        args.cur_output_dir = cur_output_dir

        # define the output dir for each target goal
        writer_result = open(os.path.join(cur_output_dir, "results.txt"), "w")
        writer_logger = open(os.path.join(cur_output_dir, "logger.txt"), "w")
        topk_prompt_dir = os.path.join(cur_output_dir, "topk_results")
        os.makedirs(topk_prompt_dir, exist_ok=True)

        # print the parameter
        write_log(writer_logger, "======  Current Parameter =======")
        for para in args.__dict__:
            write_log(writer_logger, para + ': ' + str(args.__dict__[para]))

        # construct the sensitive word list of the attack target
        synonym_words = synonym_model.get_synonym(cur_forbidden_words, k=args.synonym_num)
        for word in cur_forbidden_words:
            if len(word.split()) > 1:
                cur_forbidden_words.extend(word.split())
        cur_forbidden_words.extend([word[0] for word in synonym_words])
        write_log(writer_result, "the forbidden words is {}".format(cur_forbidden_words))

        init_time = time.time()
        for i in range(len(clean_prompts)):
            start_time = time.time()
            ori_object = objects[i].lower()
            ori_prompt = clean_prompts[i].strip().lower()

            write_log(writer_logger, "Start to train {}^th object: {}, "
                                     "attack target: {},  \n the input prompt: {}".format(
                                    i, ori_object, attack_tgt,
                                    ori_prompt + ' ' + ' '.join(["<|startoftext|>"] * args.add_suffix_num)))

            assert ori_object in ori_prompt, "Not match the ori object and the input prompt"
            learned_prompt, best_sim, prompt_topk = optimize_prompt(
                                                        model, preprocess, tokenizer, args, device,
                                                        ori_prompt=ori_prompt,
                                                        target_images=orig_images, forbidden_words=cur_forbidden_words,
                                                        suffix_num=args.add_suffix_num,
                                                        only_english_words=True)
            end_time = time.time()
            write_log(writer_logger, "The final prompt is {}".format(learned_prompt))
            write_log(writer_logger, "The best sim is {:.3f}".format(best_sim))
            write_log(writer_logger, "Spent time: {:.3f}s".format(end_time-start_time))
            save_top_k_results(topk_prompt_dir, ori_prompt, prompt_topk)
            writer_result.write(learned_prompt + "\n")
        finish_time = time.time()
        all_time = finish_time - init_time
        write_log(writer_logger, "Cong!! Finish the experiment of {}, spent time is {}h{}m".format(
            attack_tgt, all_time//3600, (all_time%3600)//60))
