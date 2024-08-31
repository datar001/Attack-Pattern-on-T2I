# Revealing Vulnerabilities in Stable Diffusion via Targeted Attacks

<img src=examples/framework.png  width="70%" height="40%">

## Dependencies

- PyTorch == 2.0.1
- transformers == 4.23.1
- diffusers == 0.11.1
- ftfy==6.1.1
- accelerate=0.22.0
- python==3.8.13

## Usage

1. Download the [word2id.pkl and wordvec.pkl](https://drive.google.com/drive/folders/1tNa91aGf5Y9D0usBN1M-XxbJ8hR4x_Jq?usp=sharing) for the synonym model, and put download files into the Word2Vec dir.

2. A script is provided to generate adversarial prompts by inputting the clean prompt and attack target

   ```python
   # Traning for generating the adversarial prompts
   python run.py --config_path ./config.json
   ```

3. Choose a T2I model to conduct attack experiments

   ```python
   python Attack_for_Evaluation.py --adversarial_prompt_dir [the dir path of adversarial prompts] --output_image_dir [the path of generated images] --attack_model [the attcked T2I model]
   ```

   

## Training Parameters

Config can be loaded from a JSON file. 

Config has the following parameters:

- `add_suffix_num`: the number of suffixes in the word addition perturbation strategy. The default is 5.
- `synonym_num`: The forbidden number of synonyms. The default is 10.
- `iter`: the total number of iterations. The default is 500.
- `lr`: the learning weight for the [optimizer](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html). The default is 0.1
- `weight_decay`: the weight decay for the optimizer.
- `print_step`: The number of steps to print a line giving current status
- `batch_size`: number of referenced images used for each iteration.
- `clip_model`: the name of the CLiP model. The default is `"laion/CLIP-ViT-H-14-laion2B-s32B-b79K"`.
- `prompt_path`: The path of clean prompt file.
- `forbidden_words`: A txt file for representing the forbidden words for each attack target.
- `target_path`: The file path of reference images.
- `output_dir`: The path for saving the learned adversarial prompts.

## Citation

If you find the repo useful, please consider citing.

```

```

