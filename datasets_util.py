import csv

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from transformers import CanineTokenizer, CanineModel

import torch
import torch.nn.functional as F

def preprocess_string(s):
    return ' '.join(s.replace('\n','')
                    .replace('\t','')
                    .split())

class QueryContentDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer =AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')#AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2") #

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        query = [x[0] for x in data]
        context = [x[1] for x in data]
        labels = [int(x[2])-1 for x in data]

        queryEncdoing = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True)
        contextEncoding = self.tokenizer(context, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(queryEncdoing['input_ids'])
        attention_mask = torch.LongTensor(queryEncdoing['attention_mask'])
        #token_type_ids = torch.LongTensor(queryEncdoing['token_type_ids'])

        token_ids2 = torch.LongTensor(contextEncoding['input_ids'])
        attention_mask2 = torch.LongTensor(contextEncoding['attention_mask'])
        #token_type_ids2 = torch.LongTensor(contextEncoding['token_type_ids'])

        labels = torch.LongTensor(labels)

        return (token_ids, attention_mask,
                token_ids2, attention_mask2,
                labels)

    def collate_fn(self, all_data):
        (token_ids,  attention_mask,
         token_ids2, attention_mask2,
         labels) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'attention_mask_2': attention_mask2,
                'labels': labels,
            }

        return batched_data
    
class ToxicityDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer =CanineTokenizer.from_pretrained(args.tokenizer,use_fast=False)#AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")## #

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        query = [x[0] for x in data]
        labels = [int(x[1]) for x in data]

        queryEncdoing = self.tokenizer(query, return_tensors='pt',padding=True, max_length=196, truncation=True)

        token_ids = torch.LongTensor(queryEncdoing['input_ids'])
        attention_mask = torch.LongTensor(queryEncdoing['attention_mask'])
        #token_type_ids = torch.LongTensor(queryEncdoing['token_type_ids'])

        labels = torch.LongTensor(labels)

        return (token_ids, attention_mask,
                labels)

    def collate_fn(self, all_data):
        (token_ids,  attention_mask,
         labels) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'attention_mask_1': attention_mask,
                'labels': labels,
            }

        return batched_data
    
    



class ToxicityDatasetMSE(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer =AutoTokenizer.from_pretrained(args.tokenizer,use_fast=False,cache_dir = 'cache')#
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        query = [x[0] for x in data]
        labels = [float(x[1]) for x in data]

        queryEncdoing = self.tokenizer(query, return_tensors='pt',padding=True, truncation=True, max_length=128)

        token_ids = torch.LongTensor(queryEncdoing['input_ids'])
        attention_mask = torch.LongTensor(queryEncdoing['attention_mask'])
        #token_type_ids = torch.LongTensor(queryEncdoing['token_type_ids'])

        labels = torch.FloatTensor(labels)

        return (token_ids, attention_mask,
                labels)

    def collate_fn(self, all_data):
        (token_ids,  attention_mask,
         labels) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'attention_mask_1': attention_mask,
                'labels': labels,
            }

        return batched_data
    
    
    
class ToxicityDatasetMSEInference(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer =AutoTokenizer.from_pretrained('microsoft/deberta-v3-base',use_fast=False,cache_dir = 'cache')#AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2") #

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        query = [str(x[1]) for x in data]
        id_url = [str(x[0]) for x in data]

        queryEncdoing = self.tokenizer(query, return_tensors='pt',padding=True, truncation=True, max_length=512)

        token_ids = torch.LongTensor(queryEncdoing['input_ids'])
        attention_mask = torch.LongTensor(queryEncdoing['attention_mask'])
        #token_type_ids = torch.LongTensor(queryEncdoing['token_type_ids'])


        return (token_ids, attention_mask,id_url)

    def collate_fn(self, all_data):
        (token_ids,  attention_mask,id_url) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'attention_mask_1': attention_mask,
                'id_url': id_url
            }

        return batched_data  






def load_queryContext_data(file_name, split='train'):
    num_labels = set()
    queryContext_data = []
    if split == 'test':
        with open(file_name, 'r') as fp:
            reader = csv.reader(fp,delimiter = '\t')
            for record in reader:
                queryContext_data.append((record[0],
                                        record[1],
                                        record[2]))
                num_labels.add(record[2])
    else:
        with open(file_name, 'r') as fp:
            reader = csv.reader(fp,delimiter = '\t')
            for record in reader:
                queryContext_data.append((record[0],
                                        record[1],
                                        record[2]))
                num_labels.add(record[2])

    return queryContext_data, num_labels





def load_toxic_data_mse(file_name, split='train'):
    toxic_data = []
    if split == 'test':
        with open(file_name, 'r') as fp:
            reader = csv.reader(fp,delimiter = '\t')
            for record in reader:
                toxic_data.append((preprocess_string(record[1]),
                                        record[2]))
    else:
        with open(file_name, 'r') as fp:
            reader = csv.reader(fp,delimiter = '\t')
            for record in reader:
                #if len(toxic_data) < 500:
                toxic_data.append((preprocess_string(record[1]),
                                        record[2]))
    #print(labels)
    return toxic_data, {}


def load_toxic_data_mse_inference(file_name, split='train'):
    toxic_data = []
    with open(file_name, 'r') as fp:
        reader = csv.reader(fp,delimiter = '\t')
        for record in reader:
            #if len(toxic_data) < 500:
            toxic_data.append((record[0],
                                    preprocess_string(record[1])))
    #print(labels)
    return toxic_data, {}

