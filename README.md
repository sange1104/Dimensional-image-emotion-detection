# Dimensional-image-emotion-detection 
This repository provides the implementation code for the paper **Enhancing dimensional image emotion detection with low-resource dataset via two-stage training**. This work proposes a new method to train the model to predict Valence-Arousal emotion values with using categorical dataset, where each image is labeled with emotion categories.


Overview✨
------------------------ 
* [scripts/]() contains code for implementing the model in the paper.
* [checkpoints/]() is a repository where a checkpoint of the trained model such as weight information or optimizer state would be saved. 
* README.md
* requirements.txt

Environmental setup
------------------------
For experimental setup, ``requirements.txt`` lists down the requirements for running the code on the repository. Note that a cuda device is required.
The requirements can be downloaded using,
```
pip install -r requirements.txt
``` 

Usage
------------------------
1. Download dataset and split into train, val, and test set.
Information about dataset we used in our experiments are as below.


2. Set up the object_detection folder. We used pre-trained object detection model from the [Faster R-CNN with model pretrained on Visual Genome](https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome). For the object_detection folder, you better follow the guideline of [this repository](https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome).

3. You can run [train.py]() setting arguments as follows: 
<!-- * gpu_num: required, int, no default
* iters: not required, int, 500
* lambda_1: not required, float, 0.9
* lambda_2: not required, float, 0.99
* latent_dim: not required, int, 100
* lr_g: not required, float, 1e-04
* iters: not required, float, 1e-04
* batch_size: not required, int, 32  -->

|Name|Required|Type|Default|
|---|---|---|---|
|gpu_num|Yes|int|-|
|iters|No|int|500|
|lambda_1|No|float|0.9|
|lambda_2|No|float|0.99|
|latent_dim|No|int|100|
|lr_g|No|float|1e-04|
|lr_d|No|float|1e-04| 
|batch_size|No|int|32| 


You can train the model as follows:
```
python ./scripts/train.py --gpu_num 0 --iters 500
```  

4. The checkpoints of the best validation performance will be saved in [checkpoints]() directory. You can further train model or use for inference with this checkpoint.
