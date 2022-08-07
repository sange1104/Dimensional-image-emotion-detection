from torchvision import models
import torch.nn as nn

class SingleTaskResnet(nn.Module):
    def __init__(self, feature_dim, class_num):
        super(SingleTaskResnet, self).__init__()
        
        modules = list(models.resnet101(pretrained=True).children())[:-1] 
        self.resnet = nn.Sequential(*modules)
#         for child in self.resnet.children():
#             for param in child.parameters():
#                 param.requires_grad = False  
        self.resnet_features_dim = 2048  
        self.relu = nn.ReLU()
        self.FC_va = nn.Linear(self.resnet_features_dim, class_num)   
        
    def forward(self, image): 
        x = self.resnet(image)    
        x = x.flatten(1) # batch, 2048   
        x = self.relu(x) 
        x = self.FC_va(x)       
        return x        
    
class MultiTaskResnet(nn.Module):
    def __init__(self, feature_dim, class_num, va_dim):
        super(MultiTaskResnet, self).__init__()
        
        modules = list(models.resnet101(pretrained=True).children())[:-1] 
        self.resnet = nn.Sequential(*modules)
#         for child in self.resnet.children():
#             for param in child.parameters():
#                 param.requires_grad = False  
        self.resnet_features_dim = 2048  
        self.relu = nn.ReLU()
        self.FC_cat = nn.Linear(self.resnet_features_dim, class_num)   
        self.FC_va = nn.Linear(self.resnet_features_dim, va_dim)   
        
    def forward(self, image): 
        x = self.resnet(image)    
        x = x.flatten(1) # batch, 2048   
        x = self.relu(x) 
        y_cat = self.FC_cat(x)   
        y_va = self.FC_va(x)        
        return y_cat, y_va            

