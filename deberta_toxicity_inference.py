import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
#from sklearn.manifold import TSNE
import torch
from torch.utils.data import Dataset, DataLoader
import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace
import csv
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score, precision_score, recall_score,mean_squared_error, mean_absolute_error

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModel

# change it with respect to the original model
from tqdm import tqdm
import math

import torch
import gc
torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.get_device_name(0)
device = torch.cuda.current_device()
torch.cuda.empty_cache()
gc.collect()


MPNET_HIDDEN_SIZE =768
N_CLASSES = 1


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class FactCheckClassifier(torch.nn.Module):

    def __init__(self, config):
        super(FactCheckClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.model =  AutoModel.from_pretrained('microsoft/deberta-v3-base',cache_dir = 'cache')
        self.toxicity_agn = AutoModel.from_pretrained('microsoft/deberta-v3-base',cache_dir = 'cache')
        for  param in self.model.parameters():
            param.requires_grad = True
            
        for  param in self.toxicity_agn.parameters():
            param.requires_grad = True
        self.temperature = 0.07   
        self.scale = math.sqrt(MPNET_HIDDEN_SIZE)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.factcheck_head = torch.nn.Linear(MPNET_HIDDEN_SIZE*2,MPNET_HIDDEN_SIZE)
        self.factcheck_head.requires_grad = True
        self.average_factcheck_head = torch.nn.Linear(MPNET_HIDDEN_SIZE*2,MPNET_HIDDEN_SIZE)
        self.average_factcheck_head.requires_grad = True
        self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.linear = torch.nn.Linear(MPNET_HIDDEN_SIZE, N_CLASSES)
        self.linear.requires_grad = True
        self.relu = torch.nn.ReLU()
        self.batchnorm_final = torch.nn.BatchNorm1d(MPNET_HIDDEN_SIZE)
        self.batchnorm_final.requires_grad = True
        self.attention_linear = torch.nn.Linear(MPNET_HIDDEN_SIZE,MPNET_HIDDEN_SIZE)
        self.attention_linear.requires_grad = True
        self.loss = torch.nn.MSELoss()
        self.relu2 = torch.nn.ReLU()

    def attention(self, inputs, query):
        sim = torch.einsum('blh,bh->bl', inputs, query) / self.scale  # (B, L)
        att_weights = torch.nn.functional.softmax(sim, dim=1)  # (B, L)
        context_vec = torch.einsum('blh,bl->bh', inputs, att_weights)  # (B,H)
        return context_vec
    
    def get_mask(self,labels):
        # Check that i, j and k are distinct
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        
        distinct_indices = i_not_equal_j
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        i_equal_j = label_equal.unsqueeze(2)
        valid_labels =  i_equal_j
        return distinct_indices & valid_labels
    
    def get_loss(self,input_ids_1, attention_mask_1,b_labels):
        greater_than_0_5 = b_labels > 0.5
        has_true = greater_than_0_5.any()
        has_false = (~greater_than_0_5).any()
        #if has_true and has_false:
        tag_emebddings = self.toxicity_agn(input_ids_1, attention_mask_1)['last_hidden_state'][:,0,:]
        tag_emebddings = self.dropout(tag_emebddings)
        dot = torch.mm(tag_emebddings, tag_emebddings.t())
        #
        #print(greater_than_0_5)
        mask = self.get_mask(greater_than_0_5.int().clone().detach()).squeeze().to(device)
        square_norm2 = torch.diag(dot)
        bottom = torch.sqrt(square_norm2).unsqueeze(0)*torch.sqrt(square_norm2).unsqueeze(1)
        bottom_mask =  torch.zeros(tag_emebddings.shape[0], tag_emebddings.shape[0])
        bottom_mask =1 -bottom_mask.fill_diagonal_(1).to(device)
        #triplet_loss[triplet_loss < 0] = 0
        #print(torch.sum(bottom_mask *torch.exp(dot/(self.temperature*bottom)),axis=1))
        top = torch.sum(mask*torch.exp(dot/(self.temperature*bottom)),axis=1)+1e-16
        bottom = torch.sum(bottom_mask *torch.exp(dot/(self.temperature*bottom)),axis=1)+1e-16
        loss = -1*torch.log(top/bottom)
        #loss[loss < 0] = 0= 
        #print(loss)

        deberta_emebddings = self.model(input_ids_1, attention_mask_1)['last_hidden_state'][:,0,:]
        deberta_emebddings= self.dropout(deberta_emebddings)
        deberta_emebddings = torch.squeeze(deberta_emebddings)
        tag_aware_attention = self.attention(deberta_emebddings.unsqueeze(1),self.attention_linear(tag_emebddings))
        output1 = torch.cat((tag_aware_attention,deberta_emebddings), axis = -1)
        output1 = self.average_factcheck_head(output1)
        output1= self.relu(output1)
        output1= self.batchnorm_final(output1)
        output1 = self.linear(output1)
        output1 = self.relu2(output1)
        #print(output1)
        loss2 = F.mse_loss(output1, b_labels.unsqueeze(1), reduction='sum') #/ args.batch_size
        #print(loss2)
        return loss.sum()+loss2, True
        #else:
        #    return 0, False
    
    def predict_factcheck(self,
                           input_ids_1, attention_mask_1,):
        deberta_emebddings = self.model(input_ids_1, attention_mask_1)['last_hidden_state'][:,0,:]
        deberta_emebddings = torch.squeeze(deberta_emebddings)
        tag_emebddings = self.toxicity_agn(input_ids_1, attention_mask_1)['last_hidden_state'][:,0,:]
        tag_aware_attention = self.attention(deberta_emebddings.unsqueeze(1),self.attention_linear(tag_emebddings))
        output1 = torch.cat((tag_aware_attention,deberta_emebddings), axis = -1)
        output1 = self.average_factcheck_head(output1)
        output1= self.relu(output1)
        output1= self.batchnorm_final(output1)
        output1 = self.linear(output1)
        output1 = self.relu2(output1)
        return output1
    
from datasets_util import load_queryContext_data, QueryContentDataset,load_toxic_data_mse, ToxicityDatasetMSE
class Object(object):
    pass
args = Object()
args.seed = 11711
args.epochs = 20
args.batch_size = 64
args.hidden_dropout_prob = 0.30
args.lr = 1e-5 #1e-4 for non-finetune examples, 1e-5 for finetuning 
args.tokenizer = 'microsoft/deberta-v3-base'
dev_toxic_data, num_labels= load_toxic_data_mse('deepak_comment_ratings_scale.csv', split ='dev')
dev_data = ToxicityDatasetMSE(dev_toxic_data, args)

dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=args.batch_size,
                                collate_fn=dev_data.collate_fn)
saved = torch.load('pretrain-50-1e-05-deberta-base-toxicity-training-batchnorm-mse-w-contrastive-updated-20240320.pt0')
config = saved['model_config']

model = FactCheckClassifier(config)
model.load_state_dict(saved['model'])
model = model.to(device)
#print(f"Loaded model to test from {args.filepath}")
#test_model_multitask(args, model, device)


dev_toxic_data, num_labels= load_toxic_data_mse('deepak_comment_ratings_scale.csv', split ='dev')
dev_data = ToxicityDatasetMSE(dev_toxic_data, args)

dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=args.batch_size,
                                collate_fn=dev_data.collate_fn)



#def model_eval(dataloader, model, device):
model = model.eval()  # switch to eval model, will turn off randomness like dropout
TQDM_DISABLE=False
y_true = []
y_pred = []
with torch.no_grad():
    for step, batch in enumerate(tqdm(dev_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        (b_ids1, b_mask1,
         b_labels) = (batch['token_ids_1'], batch['attention_mask_1'],
                      batch['labels'])

        b_ids1 = b_ids1.to(device)
        b_mask1 = b_mask1.to(device)

        logits = model.predict_factcheck(b_ids1, b_mask1)
        preds = logits.flatten().cpu().numpy()
        #preds = np.argmax(y_hat, axis=1).flatten()
        b_labels = b_labels.flatten().cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(b_labels)


#return mae, y_pred, y_true