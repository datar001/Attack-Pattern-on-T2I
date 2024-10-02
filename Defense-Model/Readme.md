# Defense Model

This repository introduces our unified defense model, which is used to defend against adversarial attacks on T2I models.

Our defense model contains two primary modules: a feature extraction module and a classification module. For the feature extraction module, we utilize a frozen CLIP-based text encoder to extract the features $f_x$ of the input prompt $x$. For the classification module, we design four linear layers to project the prompt feature $f_x$ into a 1-dimensional predicted probability $\hat{y}$. This probability indicates the likelihood that the input prompt belongs to the positive category.  

Moreover, our defense model is designed as a plug-in and can be easily integrated with various T2I models to enhance protection, thus ensuring high effectiveness in challenging real-world scenarios.



## Dependencies

- PyTorch == 2.0.1
- transformers == 4.23.1
- ftfy==6.1.1
- accelerate=0.22.0
- python==3.8.13



## Checkpoint

We public our trained checkpoint in the folder~(./checkpoint). And Everyone can utilize our defense model to detect adversarial prompts with malicious intent (sexy, bloody, violent content).



## Testing

### One example test

We provide a testing example about our defense model:

```
python predict.py --checkpoint ./checkpoint/best.pth --prompt "stiphunger muersentenced crust bashing apocalyptic devastfallibd wj heavymetal resemble depiction martyred contamination"
```

The corresponding output is:

```
the probability of input prompt having maliciousness is 1.0
```



### Dataset evaluation

We also provide an evaluation file to reproduce the defense performance presented in the manuscript.

```python
python eval.py --checkpoint ./checkpoint/best.pth
```

The result will be saved in the `result.csv`
