# Dimensional-image-emotion-detection 
This repository provides the implementation code for the paper **Enhancing dimensional image emotion detection with low-resource dataset via two-stage training**. This work proposes a new method to train the model to predict Valence-Arousal emotion values with using categorical dataset, where each image is labeled with emotion categories.

> **Abstract**
Image emotion analysis has gained considerable attention owing to the growing importance of computationally modeling human emotions. Most previous studies have focused on classifying the feelings evoked by an image into pre-defined emotion categories. Compared with these categorical techniques which cannot address the ambiguity and complexity of human emotions, recent studies have taken dimensional approaches to address these problems. However, there is still a limitation in that the number of dimensional datasets is significantly smaller for model training, compared with many available categorical datasets. In this paper, we proposed four types of frameworks that use categorical datasets to predict emotion values for a given image in the valence–arousal (VA) space. Specifically, our proposed framework was trained to predict continuous emotion values under the supervision of categorical labels. Extensive experiments demonstrated that our approach showed a positive correlation with the actual VA values of the dimensional dataset. In addition, our framework improved further when a small number of dimensional datasets were available for the fine-tuning
process.

The image below illustrates the model architecture.
<img width="650" alt="framework_lowresolution" src="https://user-images.githubusercontent.com/63252403/183283121-001c2914-6987-45e5-b70c-3abfc41569d9.PNG">


Overview✨
------------------------ 
* [src/]() contains code for implementing the model in the paper.
* [checkpoints/]() is a repository where a checkpoint of the trained model such as weight information or optimizer state would be saved. 
* [data/]() is a repository where different datasets exist. 
* README.md
* requirements.txt


Usage
------------------------
1. Clone our repository.

2. Prepare experimental setup. ``requirements.txt`` lists down the requirements for running the code on the repository. Note that a cuda device is required.
The requirements can be downloaded using,
```
pip install -r requirements.txt
``` 

3. Download dataset and split into train, val, and test set.
Information about datasets we used in our experiments are described as below.
* **[FI](https://arxiv.org/abs/1605.02677)**
* **[Flickr](https://ieeexplore.ieee.org/document/7472195)**
* **[Instagram](https://ieeexplore.ieee.org/document/7472195)**
* **[FlickrLDL](https://ojs.aaai.org/index.php/AAAI/article/view/10485)**
* **[TwitterLDL](https://ojs.aaai.org/index.php/AAAI/article/view/10485)**
* **[Emotion6](https://ieeexplore.ieee.org/document/7298687)**

For those datasets which do not provide validation information, we arbitraily split the validation set from training set at the training code. Finally, the data directory is expected to be consisted as follows. 

```bash
data
├── FI
│   ├── train 
│   ├── val
│   └── test
├── Flickr
│   ├── train
│   ├── val
│   └── test
├── Instagram
│   ├── train
│   ├── val
│   └── test
├── FlickrLDL
│   ├── train 
│   └── test
├── TwitterLDL
│   ├── train 
│   └── test
└── Emotion6
    ├── train 
    └── test
```

In addition, [NRC-VAD lexicon file](https://aclanthology.org/P18-1017/) is also needed for label conversion process. This file is required to be located in [data/]() directory.

4. You can run [train.py]() setting arguments as follows: 
* **gpu_num**: required, int
* **dataname**: required, str, options = [FI, Flickr, Instagram, FlickrLDL, TwitterLDL]
* **mode**: required, str, options = [one_stage, two_stage]
* **task**: required, str, options = [single, multi]
* **lr**: not required, float, 1e-05
* **decay**: not required, float, 1e-05
* **batch_size**: not required, int, 10
* **early_stop**: not required, int, 5
* **num_epochs**: not required, int, 100 
* **seed**: not required, int, 42

You can train the model as follows: 
```
python ./src/train.py --gpu_num 0 --dataname FI --mode one_stage --task single 
```  

5. The checkpoints of the best validation performance will be saved in [checkpoints]() directory. You can further train model or use for inference with this checkpoint.
