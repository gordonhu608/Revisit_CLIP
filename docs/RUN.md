# Training and Evaluation

We provide bash scripts in [scripts/](../scripts) for our work.
Make sure to configure the dataset paths in environment variable `DATA` and run the commands from the main directory `Revisit_CLIP/`.


### Training time and compute
We train our model on each dataset with a batch size equal to the shot size using a **single** RTX 3080 GPU.

#### (1) Few shot training setting
The default training settings are provided in config file at `configs/trainers/Our/vit_b16.yaml`. All hyper-parameters such as prompt length, prompt depth, etc., can be modified using this config file.

Below, we provide instructions to train our method on eurosat. 


```bash
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397]

# seed=1
# trains and evaluates on all classes
bash scripts/ours/few_shot_train.sh eurosat 1

# seed=2
# trains and evaluates on all classes
bash scripts/ours/few_shot_train.sh eurosat 2

# seed=3
# trains and evaluates on all classes
bash scripts/ours/few_shot_train.sh eurosat 3
```

#### Training and Evaluating other variants

For other variants including vision, language prompting techniques, we provide their corresponding configs and scripts as follows.

```
configs
|–– datasets/
|–– trainers/
|   |–– CoCoOp/
|   |–– CoOp/
|   |–– MaPLe/
```

```
scripts
|–– cocoop/
|–– coop/
|–– maple/
```

Please use the corresponding config and script files and follow the same instructions for our method as provided in order to train and evaluate the other variants. Same instructions can be followed to reproduce results of other variants using provided pretrained weights.
