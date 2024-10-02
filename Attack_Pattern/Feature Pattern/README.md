# Crafting Adversarial Prompts Using Feature Pattern

This code is a variant of the model-independent attack framework, aiming to generate adversarial prompts that align with malicious images in the CLIP embedding space. The alignment is defined as the condition where the similarity between two features surpasses a specified threshold. This code set the threshold as 0.4.

- input: malicious images (We list 10 malicious images used in our manuscript in "./reference_images")
- output: adversarial prompts

## Dependencies

- PyTorch == 2.0.1
- transformers == 4.23.1
- ftfy==6.1.1
- accelerate=0.22.0
- python==3.8.13

## Usage

1. Download the [word2id.pkl and wordvec.pkl](https://drive.google.com/drive/folders/1tNa91aGf5Y9D0usBN1M-XxbJ8hR4x_Jq?usp=sharing) for the synonym model, and put download files into the Word2Vec dir.

2. A script is provided to generate adversarial prompts aligning with malicious images in CLIP embedding space

   ```python
   # Traning for generating the adversarial prompts
   python run.py --config_path ./config.json
   ```

   The generated adversarial prompts will be saved in "./result".

   

   

## Training Parameters

Config can be loaded from a JSON file. 

The important config has the following parameters:

- `token_num`: The token number of the generated adversarial prompt. The default is 16.
- `sim_thres`: The threshold defined in the feature alignment. The default is 0.4
- `synonym_num`: The forbidden number of synonyms. The default is 5.
- `iter`: the total number of iterations. The default is 500.
- `clip_model`: the name of the CLiP model. The default is `"laion/CLIP-ViT-H-14-laion2B-s32B-b79K"`.
- `forbidden_words`: A txt file for representing the forbidden words for each attack target.
- `target_path`: The file path of reference images.
- `output_dir`: The path for saving the learned adversarial prompts.

