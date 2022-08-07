import torch
import pandas as pd

class ConvertLogitsToClasses(nn.Module):
    def __init__(self, dataname, device, lexicon_path = '../data/NRC-VAD-Lexicon.txt'):
        super(ConvertLogitsToClasses, self).__init__()
        self.dataname = dataname
        self.device = device
        self.lexicon = pd.read_csv(lexicon_path, sep="\t",header=None)
        self.lexicon.columns = ['word', 'v', 'a', 'd']
        # scaling to 1~9
        self.lexicon.iloc[:, 1:] = (self.lexicon.iloc[:, 1:] * 8) + 1
        
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6).to(self.device)
         
        if self.dataname == 'flickr': 
            label_names = ['positive', 'negative', 'neutral']
        elif self.dataname == 'FI': 
            label_names = ['amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness']
        elif self.dataname == 'instagram': 
            label_names = ['positive', 'negative', 'neutral']
        elif self.dataname == 'FlickrLDL': 
            label_names = ['amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness']
        elif self.dataname == 'TwitterLDL': 
            label_names = ['amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness']
            
        self.emo_cates_vad = []
        for label_str in label_names:
            row = self.lexicon[self.lexicon.word==label_str] 
            v, a, _ = float(row['v']), float(row['a']), float(row['d']) 
            self.emo_cates_vad.append([v, a])
        self.emo_cates_vad = torch.FloatTensor(self.emo_cates_vad).to(self.device) # (8, 2)
        
    def forward(self, logits):
        assert len(logits.shape) == 2 # shape = (batch, 2) 
        # calculate similarites
        total_cos = []
        for i in logits:
            total_cos.append(self.cos_sim(i.unsqueeze(0), self.emo_cates_vad)) # (1, 2), (8, 2)
        total_cos = torch.stack(total_cos) 
        
        if self.dataname in ['FlickrLDL', 'TwitterLDL']:
            ones, zeros = torch.ones(total_cos.shape).to(self.device), torch.zeros(total_cos.shape).to(self.device)
            pred_labels = torch.where(total_cos>0.5, ones, zeros)  
        else:
            pred_labels = torch.argmax(total_cos, dim=1) 
        return pred_labels