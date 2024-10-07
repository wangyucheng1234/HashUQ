# HashUQ
Implementation of paper: Hashing with Uncertainty Quantification via Sampling-based Hypothesis Testing. We implement our model based on [DeepHash-pytorch](https://github.com/swuxyj/DeepHash-pytorch).

# Environment

```
torch==1.8.1+cu111
lap==0.4.0
matplotlib==3.4.2
torchvision==0.9.1+cu111
scipy==1.6.3
numpy==1.19.5
tqdm==4.61.0
pandas==1.2.4
Flask==3.0.0
Pillow==10.1.0
scikit_learn==1.3.1
```

# Dataset Preparation
Please find the instructions to setup datasets in [DeepHash-pytorch](https://github.com/swuxyj/DeepHash-pytorch). Notice that the specfic split we use for NUS-WIDE dataset is NUS-WIDE-21. 
After downloading the dataset, please unzip the dataset at
```
../image_hashing_data/DATA_SET
```

where DATA_SET represent the specfic dataset we want to test, and need to be replaced by ``imagenet``, ``coco`` or ``nuswide_21``.

We include our dataset split in ```./data/```. The split is constructed by randomly select 10% of the data samples in the original training split to do the validation and model selection using ```Devide_TrainingSet_ImageNet```, ```Devide_TrainingSet_MSCOCO``` and ```Devide_TrainingSet_NUSWIDE```. Please move the txt files to the root folder of each dataset. 


# Training
To train our model with ``Center-Target'' construction on ImageNet, use the command:
```
python CSQ_HashUQ.py --grad_est cf --no-pairwise --no-tqdm --dataset imagenet --KL_regularization 1 --val --rt_st hamuct --sample_method MCD --sample 100
```

You can replace the argument ``imagenet`` to ``coco`` or ``nuswide_21`` to train and check the performance on other dataset. 

Model will be automaticly saved at 
```
./saved_model/
```

# Testing
To test our model with ``Center-Target'' construction on ImageNet after trai, use the command:
```
python CSQ_HashUQ.py --grad_est cf --no-pairwise --no-tqdm --dataset imagenet --KL_regularization 1 --val --rt_st hamuct --sample_method MCD --sample 100 --no-train
```