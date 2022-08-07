from tqdm import tqdm
import torch
import torch.nn as nn  
import os 
from collections import defaultdict 
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter 
import argparse 
from sklearn.metrics import jaccard_score
import pandas as pd
from scipy import stats 
from utils import ConvertLogitsToClasses
from model import SingleTaskResnet, MultiTaskResnet
from dataset import load_ctg_dataloader, load_dim_dataloader, load_multi_dataloader, load_va_dataloader

class Trainer(): 
    def __init__(self, dataname, lr, decay, early_stop, batch_size, random_seed, num_epochs=100): 
        self.device = self.set_device()
        self.set_loss_fn()
        self.num_epochs = num_epochs
        self.learning_rate = lr 
        self.weight_decay = decay 
        if dataname in ['FI', 'FlickrLDL', 'TwitterLDL']:
            self.class_num = 8
        elif dataname in ['flickr', 'instagram']:
            self.class_num = 3
        self.alpha = 0.5
        self.hist = defaultdict(list)
        self.save_dir = '../checkpoints'
        self.best_loss = float("inf")
        self.best_v = 0
        self.scheduling = True
        self.dataname = dataname 
        self.patience = 0
        self.early_stop = early_stop
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.converter = ConvertLogitsToClasses(self.dataname, self.device)

    def set_device(self):
        if torch.cuda.is_available():  
            print('There are %d GPU(s) available.'%torch.cuda.device_count())
            print('We will use the GPU: %d'%gpu_num)
            return torch.device("cuda:0")
        else: 
            print('No GPU available, using the CPU instead.')
            return torch.device("cpu") 

    def get_params(self, model, feature_extract=False): 
        if feature_extract:
            freeze_names = ['resnet.0', 'resnet.4', 'resnet.5', 'resnet.6', 'resnet.7', 'FC_1', 'classifier'] 
            
            for name,param in model.named_parameters():
                if freeze_names[feature_extract] in name: 
                    break
                else:
                    param.requires_grad = False 

        params_to_update = [p for p in model.parameters() if p.requires_grad]
        train_params, total_params = sum([p.numel() for p in model.parameters() if p.requires_grad]), sum(p.numel() for p in model.parameters()) 
        print('Trainable params # %d, Total params # %d'%(train_params, total_params))
        return params_to_update

    def set_optimizer(self, params): 
        optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40,50], gamma=0.5)
        return optimizer, scheduler
    
    def set_loss_fn(self):
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.bce_loss_fn = nn.BCELoss()
        self.mse_loss_fn = nn.MSELoss() 

    def build_model(self, task):
        hidden_dim = 8
        va_dim = 2
        
        if task == 'single':
            model = SingleTaskResnet(hidden_dim, va_dim).to(self.device)
        elif task == 'multi':
            model = MultiTaskResnet(hidden_dim, self.class_num, va_dim).to(self.device)
        return model
         
    def compute_multi_loss(self, pred_vads, pred_labels, vads, labels, cls_type='single') :
        # regression loss
        reg_loss = self.mse_loss_fn(pred_vads, vads)
        
        # cross-entropy loss
        if cls_type == 'single':
            ce_loss = self.ce_loss_fn(pred_labels, labels) 
        elif cls_type == 'multilabel':
            ce_loss = self.bce_loss_fn(pred_labels, labels.type(torch.FloatTensor).to(self.device))  
        
        # total loss
        total_loss = ce_loss * (1 - self.alpha) + reg_loss * self.alpha
        return total_loss
        
    def backward_step(self, loss, optimizer, scheduler):
        # calculate loss
        loss.backward()
        
        # update params
        optimizer.step()
        if self.scheduling:
            scheduler.step()
            
        # initialize
        optimizer.zero_grad()
        
    def compute_acc(self, preds, labels):  
        assert len(preds) == len(labels)
        if self.dataname in ['FlickrLDL', 'TwitterLDL']: 
            score = jaccard_score(preds.cpu(), labels.cpu(), average='weighted')
            return score
        return torch.sum(preds.data == labels.data).item() / len(preds)
    
    def _init_epoch_info(self):  
        total_losses = 0.
        return total_losses
        
    def _cum_epoch_info(self, loss, total_losses):  
        total_losses += loss.item()
        return total_losses
    
    def _cum_hist(self, train_loss, val_loss, val_acc, val_v, val_a, test_acc=None, test_v=None, test_a=None):
        self.hist['train_loss'].append(train_loss) 
        self.hist['val_loss'].append(val_loss)  
        self.hist['val_acc'].append(val_acc) 
        self.hist['val_v'].append(val_v)  
        self.hist['val_a'].append(val_a) 
        self.hist['test_acc'].append(test_acc)
        self.hist['test_v'].append(test_v)  
        self.hist['test_a'].append(test_a) 
        
    def save_checkpoint(self, model, save_name):
        path = os.path.join(self.save_dir, save_name)
        torch.save(model.state_dict(), path)
        print("Saving Model to:", path, "...Finished.")
    
    def load_checkpoint(self, model, save_name): 
        path = os.path.join(self.save_dir, save_name)
        model.load_state_dict(torch.load(path)) 
        print("Loading checkpoint:", path, "...Finished.")

    def predict_single(self, model, dataloader, type): 
        '''
        Return
        - total loss
        - total va predictions
        - total va trues
        - total labels (if available)
        '''
        model.eval() 
        with torch.no_grad():
            total_pred_vads = []
            total_scores = []
            total_pred_cats = []
            total_labels = []
            total_losses = self._init_epoch_info()
            for batch in tqdm(dataloader):
                image = batch['image'].to(self.device) 
                scores = batch['va'].to(self.device)     

                # forward
                outputs = model(image) 

                # compute loss
                if type == 'cls': 
                    labels = batch['label'].to(self.device) 
                    total_labels.append(labels)
                    
                    if self.cls_type == 'single': 
                        pred_labels = self.converter(outputs)  
                        loss = -1 * torch.FloatTensor(1) # flag -1
                    elif self.cls_type == 'multilabel': 
                        # convert to categories
                        pred_labels = self.converter(outputs)  
                        loss = self.bce_loss_fn(pred_labels, labels.type(torch.FloatTensor).to(self.device))  
                    total_pred_cats.append(pred_labels)
                elif type == 'reg':
                    loss = self.mse_loss_fn(outputs, scores) 

                # cumulate
                total_losses = self._cum_epoch_info(loss, total_losses)
                total_pred_vads.append(outputs)
                total_scores.append(scores)
            total_losses /= len(dataloader)
            if type == 'cls':
                total_pred_cats = torch.cat(total_pred_cats, dim=0) 
                total_labels = torch.cat(total_labels, dim=0) 
        return total_losses, torch.cat(total_pred_vads, dim=0), torch.cat(total_scores, dim=0), total_pred_cats, total_labels
            
    def predict_multi(self, model, dataloader, type): 
        '''
        Return 
        - total loss
        - total va predictions
        - total va trues
        - total label predictions
        - total label trues
        '''
        model.eval() 
        with torch.no_grad():
            total_pred_vads = []
            total_vads = []
            total_pred_cats = []
            total_labels = []
            total_losses = self._init_epoch_info()

            for batch in tqdm(dataloader):
                image = batch['image'].to(self.device) 
                scores = batch['va'].to(self.device) 
                if type == 'cls':  
                    labels = batch['label'].to(self.device) 
                else:
                    labels = self.converter(scores) 

                # forward
                pred_logits, pred_vads = model(image) 
                if self.cls_type == 'single':
                    pred_labels = torch.argmax(pred_logits, dim=1)
                elif self.cls_type == 'multilabel':  
                    pred_logits = nn.Sigmoid()(pred_logits)
                    ones, zeros = torch.ones(pred_logits.shape).to(self.device), torch.zeros(pred_logits.shape).to(self.device)
                    pred_labels = torch.where(pred_logits>0.5, ones, zeros) 

                # compute loss
                loss = self.compute_multi_loss(pred_vads, pred_logits, scores, labels, self.cls_type) 

                # cumulate
                total_losses = self._cum_epoch_info(loss, total_losses)
                total_pred_vads.append(pred_vads)
                total_vads.append(scores)
                total_pred_cats.append(pred_labels)
                total_labels.append(labels)
            total_losses /= len(dataloader) 
        return total_losses, torch.cat(total_pred_vads, dim=0), torch.cat(total_vads, dim=0), torch.cat(total_pred_cats, dim=0), torch.cat(total_labels, dim=0)
             
    def evaluate_single(self, model, dataloader, type):
        '''
        evaluate 'single' task model with both cls and reg task
        type: reg or cls (data labeling type)
        - if data is FI, Flickr, Instagram, type is 'cls'
        - if data is Emotion6, type is 'reg'
        '''
        loss, logits, scores, preds, labels = self.predict_single(model, dataloader, type) 
        
        # 1) evaluate with classification task (metric: acc)
        if type == 'cls':  
            acc = self.compute_acc(preds, labels) 
        else: # reg
            acc = -1
        
        # 2) evaluate with regression task (metric: corr coef)
        coef = self.cal_coeff(logits, scores)
        return loss, acc, coef
        
    def evaluate_multi(self, model, dataloader, type):
        '''
        evaluate 'multi' task model with both cls and reg task
        type: reg or cls (data labeling type)
        - if data is FI, Flickr, Instagram, type is 'cls'
        - if data is Emotion6, type is 'reg'
        '''
        loss, logits, scores, preds, labels = self.predict_multi(model, dataloader, type)
        
        # 1) evaluate with classification task (metric: acc)
        acc = self.compute_acc(preds, labels) 
        
        # 2) evaluate with regression task (metric: corr coef)
        coef = self.cal_coeff(logits, scores)
        return loss, acc, coef
        
    def cal_coeff(self, pred, true):     
        idx_to_label = {0:'v', 1:'a'}
        idx_to_corr = {}
        for i in range(pred.shape[1]):
            r, p = stats.pearsonr(pred[:, i].cpu(), true[:, i].cpu())
            idx_to_corr[idx_to_label[i]] = (r, p)
        return idx_to_corr
    
    def train_single(self, model, data_loader, optimizer, scheduler):
        total_losses = self._init_epoch_info()
        for batch in tqdm(data_loader):
            image = batch['image'].to(self.device) 
            scores = batch['va'].to(self.device)  

            # forward
            outputs = model(image) 

            # compute loss
            loss = self.mse_loss_fn(outputs, scores)

            # backward
            self.backward_step(loss, optimizer, scheduler)

            # cumulate 
            total_losses = self._cum_epoch_info(loss, total_losses)
        total_losses /= len(data_loader) 
        return total_losses 
    
    def train_multi(self, model, data_loader, optimizer, scheduler):
        total_losses = self._init_epoch_info() 
        for batch in tqdm(data_loader):
            image = batch['image'].to(self.device) 
            scores = batch['va'].to(self.device)  
            labels = batch['label'].to(self.device)  


            # forward
            pred_logits, pred_vads = model(image) 
            if self.cls_type == 'single':
                pred_labels = torch.argmax(pred_logits, dim=1)
                loss = self.compute_multi_loss(pred_vads, pred_logits, scores, labels, self.cls_type)  
            elif self.cls_type == 'multilabel':  
                pred_labels = nn.Sigmoid()(pred_logits)
                loss = self.compute_multi_loss(pred_vads, pred_labels, scores, labels, self.cls_type)  

            # # compute loss
            # loss = self.compute_multi_loss(pred_vads, pred_logits, scores, labels, self.cls_type)  

            # backward
            self.backward_step(loss, optimizer, scheduler)

            # cumulate 
            total_losses = self._cum_epoch_info(loss, total_losses)
        total_losses /= len(data_loader) 
        return total_losses 
    
    def finetuning_multi(self, model, data_loader, optimizer, scheduler):
        total_losses = self._init_epoch_info()
        for batch in tqdm(data_loader):
            images = batch['image'].to(self.device)
            indices = batch['index'].to(self.device)  
            labels = batch['label'][indices==1.].to(self.device) 
            vads = batch['va'][indices==0.].to(self.device)     
            
            # forward
            pred_labels, pred_vads = model(images) 
            pred_labels = pred_labels[indices==1.]
            pred_vads = pred_vads[indices==0.]

            # compute loss
            loss = self.compute_multi_loss(pred_vads, pred_labels, vads, labels) 

            # backward
            self.backward_step(loss, optimizer, scheduler)

            # cumulate 
            total_losses = self._cum_epoch_info(loss, total_losses)
        total_losses /= len(data_loader) 
        return total_losses 
        
    def get_va_from_dict(self, coef_dict):
        va = []
        p_values = []
        for k in coef_dict.keys(): 
            va.append(coef_dict[k][0])
            p_values.append(coef_dict[k][1])
            print('[%s] r: %f, p-value: %f'%(k, coef_dict[k][0], coef_dict[k][1]))
        v, a = va[0], va[1]
        pval_v, pval_a = p_values[0], p_values[1]
        return v, a, pval_v, pval_a
    
    def print_results(self, it, train_loss, val_loss, val_acc, test_acc):
        print(f'Epoch {it + 1}/{self.num_epochs}')
        print('-' * 10) 
        print(f'Train loss {train_loss}')
        print(f'Val   loss {val_loss}')
        print(f'Val   acc {val_acc}')
        print(f'Test   acc {test_acc}') 
        
    def train(self, mode, task):
        assert mode in ['one_stage', 'two_stage']
        assert task in ['single', 'multi']
        self.writer = SummaryWriter('../logs/%s_%s'%(mode, task))
        
        ################
        # 1. Load data
        ################ 
        # 1) categorical dataset
        if self.dataname in ['FI', 'flickr', 'instagram']:
            self.cls_type = 'single'
            train_cat_loader, val_cat_loader, test_cat_loader = load_ctg_dataloader(self.dataname, self.batch_size)         
        elif self.dataname in ['TwitterLDL', 'FlickrLDL']:
            self.cls_type = 'multilabel'
            train_cat_loader, val_cat_loader, test_cat_loader = load_dim_dataloader(self.dataname, self.batch_size)       
        if mode == 'two_stage' and task == 'multi':
            train_loader, val_va_loader, test_va_loader, val_cat_loader, test_cat_loader = load_multi_dataloader(self.dataname, self.batch_size)

        # 2) VA dimensional dataset
        train_va_loader, val_va_loader, test_va_loader = load_va_dataloader(self.batch_size, self.dataname)
        print('Done loading data...')
                
        #####################
        # 2. Build a model
        ##################### 
        save_name = '%s_%s_%s_rs_%d.pt'%(mode, task, self.dataname, random_seed)  
        model = self.build_model(task)
        if mode == 'two_stage':
            # load the model trained with same task, but only in zero-shot mode
            # for fine-tuning
            load_name = '%s_%s_%s_rs_%d.pt'%('one_stage', task, self.dataname, random_seed) 
            self.load_checkpoint(model, load_name)
            feature_extract = 4
        else:
            feature_extract = False
        params = self.get_params(model, feature_extract) 
        optimizer, scheduler = self.set_optimizer(params)
        
        ##########################
        # 3. Train
        ##########################
        for it in range(self.num_epochs): 
            model.train()                  
            
            if task == 'single':
                if mode == 'one_stage':
                    train_loss = self.train_single(model, train_cat_loader, optimizer, scheduler)               
                    val_loss, val_acc, val_coef = self.evaluate_single(model, val_cat_loader, 'reg') 
                    _, test_acc, test_coef = self.evaluate_single(model, test_va_loader, 'reg')
                elif mode == 'two_stage':
                    train_loss = self.train_single(model, train_va_loader, optimizer, scheduler)              
                    val_loss, val_acc, val_coef = self.evaluate_single(model, val_va_loader, 'reg')        
                    _, test_acc, test_coef = self.evaluate_single(model, test_va_loader, 'reg')
                    
            elif task == 'multi':
                if mode == 'one_stage':
                    train_loss = self.train_multi(model, train_cat_loader, optimizer, scheduler)      
                    val_loss, val_acc, _ = self.evaluate_multi(model, val_cat_loader, 'cls')    
                    _, _, val_coef = self.evaluate_multi(model, val_va_loader, 'reg')       
                    _, test_acc, _ = self.evaluate_multi(model, test_cat_loader, 'cls')
                    _, _, test_coef = self.evaluate_multi(model, test_va_loader, 'reg') 
                elif mode == 'two_stage':
                    self.alpha = 0.9
                    train_loss = self.finetuning_multi(model, train_loader, optimizer, scheduler)      
                    _, val_acc, _ = self.evaluate_multi(model, val_cat_loader, 'cls')   
                    val_loss, _, val_coef = self.evaluate_multi(model, val_va_loader, 'reg')      
                    _, test_acc, _ = self.evaluate_multi(model, test_cat_loader, 'cls')
                    _, _, test_coef = self.evaluate_multi(model, test_va_loader, 'reg')
            
            # print results
            print('[Val Results]')
            val_v, val_a, _, _ = self.get_va_from_dict(val_coef)
            print('[Test Results]')
            test_v, test_a, _, _ = self.get_va_from_dict(test_coef)
            
            self.print_results(it, train_loss, val_loss, val_acc, test_acc)          
            self._cum_hist(train_loss, val_loss, val_acc, val_v, val_a, test_acc, test_v, test_a) 
            self.writer.add_scalars("%s/%s/%s"%(mode, task, dataname), {'train_loss_%s_%d'%(str(lr), self.random_seed) : train_loss, 
                'val_loss_%s_%d'%(str(lr), self.random_seed) : val_loss, 
                'val_acc_%s_%d'%(str(lr), self.random_seed) : val_acc,
                'val_v_%s_%d'%(str(lr), self.random_seed) : val_v,
                'val_a_%s_%d'%(str(lr), self.random_seed) : val_a}, it) 
              
            if val_v > self.best_v:
                self.best_v = val_v
                self.save_checkpoint(model, save_name)
                self.patience = 0
            else:
                self.patience += 1
                print('Patience %d...'%(self.patience))
            
            if self.patience > self.early_stop:
                self.save_checkpoint(model, save_name)
                break

        if self.num_epochs > 0:
            self.end_epoch = it
        
        ###############################
        # 4. Evaluate with VAD dataset
        ###############################  
        self.load_checkpoint(model, save_name) 
        if task == 'single':
            _, test_acc, _ = self.evaluate_single(model, test_cat_loader, 'cls')
            _, _, test_coef = self.evaluate_single(model, test_va_loader, 'reg')
        elif task == 'multi':
            _, test_acc, _ = self.evaluate_multi(model, test_cat_loader, 'cls')
            _, _, test_coef = self.evaluate_multi(model, test_va_loader, 'reg')

        test_v, test_a, _, _ = self.get_va_from_dict(test_coef)   
        print('[Classification] acc: %f'%(test_acc))
        self.corr_result = test_coef
        self.acc_result = test_acc   
        
def set_randomseed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__=='__main__':  
    ######################################
    # configuration
    ######################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num', type=int, required=True)
    parser.add_argument('--dataname', type=str, required=True) 
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--task', type=str, required=True) 
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=10)  
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    gpu_num = args.gpu_num
    dataname = args.dataname
    mode = args.mode
    task = args.task
    lr = args.lr
    decay = args.decay
    batch_size = args.batch_size
    early_stop = args.early_stop
    num_epochs = args.num_epochs
    random_seed = args.seed

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%gpu_num

    ######################################
    # train
    ######################################
    set_randomseed(random_seed)
    trainer = Trainer(dataname, lr, decay, early_stop, batch_size, random_seed, num_epochs)
    trainer.train(mode, task)
 
    ######################################
    # save history
    ######################################
    result_dir = '../results'
    os.makedirs(result_dir, exist_ok=True)
    history_path = os.path.join(result_dir, '%s_%s_%s_results.txt'%(mode, task, dataname))
    with open(history_path, "a") as f:
        f.write("\n")
        f.write("%d | %s | %s | %d | %d |"%(random_seed, str(lr), str(decay), batch_size, early_stop))  
        f.write("\n")
        for k in trainer.corr_result.keys():
            f.write('[%s] r: %f, p-value: %f'%(k, trainer.corr_result[k][0], trainer.corr_result[k][1]))
        f.write('[acc] r: %f'%(trainer.acc_result))
