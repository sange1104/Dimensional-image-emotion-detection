import torch
import torch.nn as nn  
import numpy as np 
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.preprocessing import MultiLabelBinarizer 
import os 
from PIL import Image     
import pandas as pd 

class CategoricalDataset(Dataset):
    def __init__(self, dataname, mode, lexicon_path = '../data/NRC-VAD-Lexicon.txt'):  
        self.dataname = dataname
        self.mode = mode
        if self.dataname == 'flickr':
            data_dir = '../data/%s/%s'%(self.dataname, self.mode)
            self.label_to_int = {j:i for i, j in enumerate(sorted(os.listdir(data_dir)))}
            self.file_list = [i for emo in os.listdir(data_dir) for i in os.listdir(os.path.join(data_dir, emo))] 
            self.name_list = [int(i.split('.')[-2].split('_')[-1]) for i in self.file_list]

        elif self.dataname == 'FI':
            data_dir = '../../Vissent/main/data/%s/%s'%(self.dataname, self.mode)
            self.label_to_int = {j:i for i, j in enumerate(sorted(os.listdir(data_dir)))}
            self.file_list = [i for emo in os.listdir(data_dir) for i in os.listdir(os.path.join(data_dir, emo))]
            self.name_list = [int(i.split('.')[-2].split('_')[-1]) for i in self.file_list]

        elif self.dataname == 'instagram':
            data_dir = '../data/%s/%s'%(self.dataname, self.mode)
            self.label_to_int = {j:i for i, j in enumerate(sorted(os.listdir(data_dir)))}
            self.file_list = [i for emo in os.listdir(data_dir) for i in os.listdir(os.path.join(data_dir, emo))]
            self.name_list = [int(i.split('.')[-2].split('_')[-2]) for i in self.file_list]
 
        self.path_list = [os.path.join(data_dir, os.path.join(i.split('_')[0], i)) for i in self.file_list]          

        # labels
        self.label_str = [i.split('/')[-2] for i in self.path_list] 
        self.labels = [self.label_to_int[i] for i in self.label_str] 
        
        # va labels
        self.lexicon = pd.read_csv(lexicon_path, sep="\t", header=None)
        self.lexicon.columns = ['word', 'v', 'a', 'd']
        # scaling to 1~9
        self.lexicon.iloc[:, 1:] = (self.lexicon.iloc[:, 1:] * 8) + 1
        # convert word to VAD score
        self.va = [torch.FloatTensor(np.array(self.lexicon[self.lexicon.word==l][['v', 'a']])[0]) for l in self.label_str] 

        input_size = 448
        self.transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def __len__(self):
        return len(self.path_list) 
    
    def __getitem__(self, item):  
        # get an image
        image_path = self.path_list[item]  
        image = Image.open(image_path).convert("RGB") 
        image = self.transforms[self.mode](image) 

        # get a label
        label = self.labels[item]

        # get va scores 
        va = self.va[item] 
         
        return {  
          'image' : image, 
          'va': va, 
          'label': label,
          'dim' : torch.zeros(8),
          'index': torch.ones(1).squeeze(),
        }

class Emotion6Dataset(Dataset):
    def __init__(self, mode, cvt_dataname = 'FI', gt_path = '../data/Emotion6/ground_truth_v2.txt', lexicon_path = '../data/NRC-VAD-Lexicon.txt'):   
        self.mode = mode   
        # image names
        data_dir = os.path.join('../data/Emotion6', self.mode)
        self.file_list = [i for emo in os.listdir(data_dir) for i in os.listdir(os.path.join(data_dir, emo))]
        self.path_list = [os.path.join(data_dir, os.path.join(i.split('_')[0], i)) for i in self.file_list]  
        
        # va scores
        self.df_label = pd.read_csv(gt_path, index_col=0)
        self.lexicon = pd.read_csv(lexicon_path, sep="\t",header=None)
        self.lexicon.columns = ['word', 'v', 'a', 'd']
        # scaling to 1~9
        self.lexicon.iloc[:, 1:] = (self.lexicon.iloc[:, 1:] * 8) + 1         
        self.va = torch.stack([torch.FloatTensor(np.array(self.df_label[self.df_label.image == f][['v', 'a']]))[0] for f in self.file_list])   # (len(data), 2)

        # convert va score to emotion category
        if cvt_dataname in ['FI']: 
            self.emo_ctg = ['amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness']
        elif cvt_dataname in ['flickr', 'instagram']:
            self.emo_ctg = ['positive', 'negative']
        elif cvt_dataname in ['FlickrLDL', 'TwitterLDL']:
            self.emo_ctg = ['amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness']
        
        # labels
        emo_ctg_va = torch.stack([torch.FloatTensor(np.array(self.lexicon[self.lexicon.word==emo][['v', 'a']])[0]) for emo in self.emo_ctg]) # (len(emo), 2)
        # emo_ctg_va = emo_ctg_va.unsqueeze(1).repeat(1, self.va.shape[0], 1) # shape = (len(emo), len(data), 2(v,a))
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        if cvt_dataname in ['FlickrLDL', 'TwitterLDL']:
            ones, zeros = torch.ones(len(self.emo_ctg)), torch.zeros(len(self.emo_ctg)) # (8,) 
            self.labels = [torch.where(self.cos_sim(i.unsqueeze(0), emo_ctg_va)>0.5, ones, zeros) for i in self.va]
        else:
            self.labels = torch.LongTensor([np.argsort([np.linalg.norm(i-j) for j in emo_ctg_va])[0] for i in self.va])
            # self.labels = [torch.argmax(self.cos_sim(i.unsqueeze(0), emo_ctg_va), dim=0) for i in self.va] 
        
        input_size = 448
        self.transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, item):   
        # get an image
        image_path = self.path_list[item]  
        image = Image.open(image_path).convert("RGB") 
        image = self.transforms[self.mode](image) 

        # get a label
        label = self.labels[item]

        # get va scores 
        va = self.va[item] 
         
        return {  
          'image' : image, 
          'va': va, 
          'label': label,
          'dim' : torch.zeros(8),
          'index': torch.zeros(1).squeeze(),
        }

class DimensionalDataset(Dataset):
    def __init__(self, dataname, mode, lexicon_path = '../data/NRC-VAD-Lexicon.txt'):  
        assert mode in ['train', 'test']
        self.dataname = dataname
        self.mode = mode     
        if self.dataname == 'FlickrLDL':
            gt_path = '../data/FlickrLDL/flickrldl_config.csv'
            data_dir = '../data/FlickrLDL/train'
        elif self.dataname == 'TwitterLDL':
            gt_path = '../data/TwitterLDL/twitterldl_config.csv'
            data_dir = '../data/TwitterLDL/train'
        
        self.file_list = os.listdir(data_dir)
        self.path_list = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
        self.df_label = pd.read_csv(gt_path) 
        
        self.emo_ctg = ['amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness']
        
        # dimension labels
        self.emo_dim = [np.array(self.df_label[self.df_label.image==i][self.emo_ctg])[0] for i in self.file_list] 
        # class labels (multi-label)
        self.labels = [np.where(i==np.max(i))[0] for i in self.emo_dim]        
        mlb = MultiLabelBinarizer()
        mlb.fit([list(range(len(self.emo_ctg)))]) 
        self.labels = [mlb.transform([label])[0] for label in self.labels]
        
        self.lexicon = pd.read_csv(lexicon_path, sep="\t",header=None)
        self.lexicon.columns = ['word', 'v', 'a', 'd']
        # scaling to 1~9
        self.lexicon.iloc[:, 1:] = (self.lexicon.iloc[:, 1:] * 8) + 1
        # va labels
        emo_ctg_va = np.stack([np.array(self.lexicon[self.lexicon.word==emo][['v', 'a']])[0] for emo in self.emo_ctg])
        self.va = [np.dot(row[np.newaxis, :], emo_ctg_va)[0]/len(self.emo_ctg) for row in self.emo_dim]
        
        input_size = 448
        self.transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def __len__(self):
        return len(self.path_list) 
    
    def __getitem__(self, item):   
        # get an image
        image_path = self.path_list[item]  
        image = Image.open(image_path).convert("RGB") 
        image = self.transforms[self.mode](image) 

        # get a label
        label = self.labels[item]

        # get va scores 
        va = self.va[item] 

        # get emotion scores 
        dim = self.emo_dim[item] 
         
        return {  
          'image' : image, 
          'va': torch.FloatTensor(va), 
          'label': label,
          'dim' : torch.FloatTensor(dim),
          'index': torch.zeros(1).squeeze(),
        }         

def load_dim_dataloader(dataname, batch_size, num_workers=2):    
    train_ds = DimensionalDataset(dataname = dataname, mode='train') 
    train_num = int(len(train_ds)*0.8)
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_num, len(train_ds)-train_num])
    test_ds = DimensionalDataset(dataname = dataname, mode='test')  
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
    return train_loader, val_loader, test_loader 

def load_ctg_dataloader(dataname, batch_size, num_workers=2):
    loaders = []
    for mode in ['train', 'val', 'test']:    
        ds = CategoricalDataset(dataname = dataname, mode=mode) 
        data_loader = DataLoader(ds, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
        loaders.append(data_loader)
    return loaders

def load_va_dataloader(batch_size, cvt_dataname, num_workers=2):    
    train_ds = Emotion6Dataset(mode='train', cvt_dataname=cvt_dataname) 
    train_num = int(len(train_ds)*0.8)
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_num, len(train_ds)-train_num])
    test_ds = Emotion6Dataset(mode='test', cvt_dataname=cvt_dataname)  
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
    return train_loader, val_loader, test_loader

def my_collate(batch):
    """Define collate_fn myself because the default_collate_fn throws errors like crazy"""
    # item: a tuple of (img, label)
    image = [item['image'] for item in batch]
    va = [item['va'] for item in batch]
    label = [item['label'] for item in batch]
    index = [item['index'] for item in batch]
    image = torch.stack(image)
    va = torch.stack(va)
    label = torch.LongTensor(label)
    index = torch.stack(index)
    return {'image':image, 'va':va, 'label':label, 'index':index}
    
def load_multi_dataloader(dataname, batch_size, num_workers=2):
    loaders = []
    cat_ds_list = [] 

    # va dataset
    train_va_ds = Emotion6Dataset(mode='train') 
    train_num = int(len(train_va_ds)*0.8)
    train_va_ds, val_va_ds = torch.utils.data.random_split(train_va_ds, [train_num, len(train_va_ds)-train_num])
    test_va_ds = Emotion6Dataset(mode='test')  
    val_va_loader = DataLoader(val_va_ds, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
    test_va_loader = DataLoader(test_va_ds, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
    
    # train  
    train_ctg_ds = CategoricalDataset(dataname = dataname, mode='train') 
    concat_ds = ConcatDataset([train_ctg_ds, train_va_ds]) 
    train_loader = DataLoader(concat_ds, batch_size=batch_size, collate_fn = my_collate, drop_last=True, shuffle=True, num_workers=num_workers)
    
    # val
    val_ctg_ds = CategoricalDataset(dataname = dataname, mode='val')  
    val_ctg_loader = DataLoader(val_ctg_ds, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
    
    # test
    test_ctg_ds = CategoricalDataset(dataname = dataname, mode='test')  
    test_ctg_loader = DataLoader(test_ctg_ds, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
    
    return train_loader, val_va_loader, test_va_loader, val_ctg_loader, test_ctg_loader
