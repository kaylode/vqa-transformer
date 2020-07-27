import torch
import torch.utils.data as data
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import string
import csv


class CSVTextClassificationDataset(data.Dataset):
    """
    Reads a CSV file.
    
    Requires first column to be text data, second column to be labels 
    """
    def __init__(self,
                 txt_dir,
                 tokenizer = str.split,
                 skip_header = True,
                 max_samples = None,
                 shuffle = False):
        
        self.dir = txt_dir
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.skip_header = skip_header
        self.max_samples = max_samples
        self.fns = self.load_txt()
        self.classes = list(set([i[1] for i in self.fns]))
        
    def load_txt(self): 
        data_list = []
        with open(self.dir, 'r', encoding = 'utf8') as fv:
            src_data = csv.reader(fv)
            if self.skip_header:
                next(src_data)
            for row in src_data:
                data_list.append(row)
        if self.shuffle:
            random.shuffle(data_list)
        data_list = data_list[:self.max_samples] if self.max_samples is not None else data_list
        return data_list
    
    def __getitem__(self, index):
        txt, label = self.fns[index]
        tokens = self.tokenizer(txt)
        
        return {"txt" : tokens,
               "label" : label} 

    def count_dict(self):
        cnt_dict = {}
        for text, label in self.fns:
            if label in cnt_dict.keys():
              cnt_dict[label] += 1
            else:
              cnt_dict[label] = 1
        return cnt_dict

    def plot(self, figsize = (8,8), types = ["freqs"]):
        cnt_dict = self.count_dict()
        ax = plt.figure(figsize = figsize)
        
        if "freqs" in types:
            plt.title("Classes Distribution")
            bar1 = plt.bar(list(cnt_dict.keys()), list(cnt_dict.values()), color=[np.random.rand(3,) for i in range(len(self.classes))])
            plt.xlabel("Classes")
            plt.ylabel("Number of samples")
            for rect in bar1:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

        if "grams" in types:
            pass


    def __len__(self):
        return len(self.fns)
    
    def __str__(self):
        s = "Custom Dataset for Text Classification \n"
        line = "-------------------------------\n"
        s1 = "Number of samples: " + str(len(self.fns)) + '\n'
        s2 = "Number of classes: " + str(len(self.classes)) + '\n'
        return s + line + s1 + s2