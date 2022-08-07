# data

This is a top directory of data folder. Data used for our experiment is required to be located here.
Download dataset and split into train, val, and test set.
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
