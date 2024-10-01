# Defense Model

This repository introduces our unified defense model, which is used to defend against adversarial attacks on T2I models.

Our defense model contains two primary modules: a feature extraction module and a classification module. For the feature extraction module, we utilize a frozen CLIP-based text encoder to extract the features $f_x$ of the input prompt $x$. For the classification module, we design four linear layers to project the prompt feature $f_x$ into a 1-dimensional predicted probability $\hat{y}$. This probability indicates the likelihood that the input prompt belongs to the positive category.  

Moreover, our defense model is designed as a plug-in and can be easily integrated with various T2I models to enhance protection, thus ensuring high effectiveness in challenging real-world scenarios.



## Training

To train our defense model, we construct an additional training and validation sets. The training set comprising 1,300 clean and malicious prompts, respectively. The validation set contains 470 clean and malicious prompts, respectively.

The training progress can be started as follows:

```python
python git_train.py --data_root ./data/ --batch_size 64 --num_epochs 25 --lr 1e-3
```

We public our trained checkpoint in the folder~(./checkpoint). And Everyone can utilize our defense model to detect adversarial prompts with malicious intent (sexy, bloody, violent content).



## Testing

We provide a testing example about our defense model:

```
python predict.py --checkpoint ./checkpoint/best.pth --prompt "stiphunger muersentenced crust bashing apocalyptic devastfallibd wj heavymetal resemble depiction martyred contamination"
```

The corresponding output is:

```
the probability of input prompt having maliciousness is 1.0
```

