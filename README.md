## Requirement

+ Create a conda virtual environment and activate it.

```
conda create --name fakeface python=3.8.13
conda activate fakeface
```

+ Install Pytorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```
conda install pytorch==1.13.1 torchvision==0.14.1 -c pytorch
```

+ Install the dependent libraries.

```
pip install -r requirements.txt
```

## Data Preparation

+ Download the five data sets respectively, unzip them to the data folder, and use the data processing code in data to process the data sets.

```
/data
    deepfake_celeb
        fake/
        true/
    deepfake_dfmnist
        fake/
        true/
    forgery
        fake/
        true/
    gan_wanghong
        fake/
        true/
    gan_yellow
        fake/
        true/
   
```

## Getting Started

### train

+ We use one Nvidia Tesla 4090 (24G) GPU for training. You need to modify the `trainval_path` in the `config.py` file before training.

```
python train.py
```

### Test

- You need to modify the `folder_name` in the `evaluate.py` file before testing.

```
python evaluate.py
```
