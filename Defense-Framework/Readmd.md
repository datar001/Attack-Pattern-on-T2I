# Defense Framework

This repository introduces our unified defense framework, which is used to defend against adversarial attacks on T2I models.

Our defense framework mainly involves a prompt detector, which is optimized by our proposed adversarial training strategy and can effectively detect adversarial prompts without impairing its functionality for legitimate user queries. Moreover, our defense framework is designed as a plug-in and can be easily integrated with various T2I models to enhance protection, thus ensuring high effectiveness in challenging real-world scenarios.



## Training

To train our defense framework, we construct an additional training set comprising 1,300 clean and malicious prompts. The training dataset will be released soon.

We public our trained checkpoint in the folder~(./checkpoint). And Everyone can utilize our defense framework to detect adversarial prompts with malicious intent~(sexy, bloody, violent content).



## Testing

We provide a testing example about our defense framework:

```
python predict.py --checkpoint ./checkpoint/best.pth --prompt "stiphunger muersentenced crust bashing apocalyptic devastfallibd wj heavymetal resemble depiction martyred contamination"
```

The corresponding output is:

```
the probability of input prompt having maliciousness is 1.0
```

