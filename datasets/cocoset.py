import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from augmentations.transforms import get_resize_augmentation, get_augmentation, Denormalize

from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

from .utils import create_masks, make_feature_batch, split_patches
from utils.utils import draw_image_caption

class CocoDataset(Dataset):
    """
    Coco dataset
    """
    def __init__(self, 
            root_dir, ann_path, 
            tokenizer, image_size=[512,512], 
            keep_ratio=False, patch_size=16, 
            type='train', cache_dir=None):

        self.cache_dir = cache_dir
        self.root_dir = root_dir
        self.ann_path = ann_path
        self.image_size = image_size
        self.patch_size = patch_size

        self.tokenizer = tokenizer
        self.transforms = A.Compose([
            get_resize_augmentation(image_size, keep_ratio=keep_ratio),
            get_augmentation(_type=type)
        ])

        self.coco = COCO(ann_path)
        self.image_ids = self.coco.getImgIds()

    def get_patch_dim(self):
        return self.patch_size * self.patch_size * 3

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        return image_path

    def load_annotations(self, image_index, return_all=False):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index])

        if not return_all:
            if len(annotations_ids)>1:
                ann_id = random.choice(annotations_ids)
            anns = self.coco.loadAnns(ann_id)[0]['caption']
        else:
            anns = self.coco.loadAnns(annotations_ids)
            anns = [i['caption'] for i in anns]
        return anns

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = self.load_image(index)
        text = self.load_annotations(index)

        return {
            'image_id': image_id,
            'image_path': image_path,
            'text': text,
        }

    def load_augment(self, image_path):
        ori_img = cv2.imread(image_path)
        image = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image /= 255.0
        image = self.transforms(image=image)['image']
        return image, ori_img

    def collate_fn(self, batch):
        
        image_paths = [s['image_path'] for s in batch]
        image_ids = [s['image_id'] for s in batch]
        
        image_names = []
        ori_imgs = []
        for image_path in image_paths:
            image_names.append(os.path.basename(image_path))

        if not self.cache_dir:
            imgs = []
            for image_path in image_paths:
                image, ori_img = self.load_augment(image_path)
                imgs.append(image)
                ori_imgs.append(ori_img)
            feats = torch.stack(imgs)
            mask_shapes = int((self.image_size[0] / self.patch_size) **2)
            image_masks = torch.ones((feats.shape[0], mask_shapes))
        else:
            npy_paths = [s[:-4] + '.npy' for s in image_names]
            npy_paths = [os.path.join(self.cache_dir, s) for s in npy_paths]
            feats = []
            for npy_path in npy_paths:
                feats.append(np.load(npy_path, allow_pickle=True)) # [64, 2048]
            try:
                feats = np.stack(feats)
                feats = torch.from_numpy(feats)
            except:
                feats = make_feature_batch(feats, pad_token=0)
            image_masks = torch.ones(feats.shape[:-1])

        texts = [s['text'] for s in batch]
        
        tokens = self.tokenizer(texts, truncation=True)
        tokens = [np.array(i) for i in tokens['input_ids']]

        texts_ = make_feature_batch(
            tokens, pad_token=self.tokenizer.pad_token_id)
        
        texts_inp = texts_[:, :-1]
        texts_res = texts_[:, 1:]

        text_masks = create_masks(
            texts_inp,
            pad_token=self.tokenizer.pad_token_id, 
            is_tgt_masking=True)
        
        texts_inp = texts_inp.squeeze(-1)

        return {
            'image_ids': image_ids,
            'image_names': image_names,
            'ori_imgs': ori_imgs,
            'image_patches': feats,
            'image_masks': image_masks.long(),
            'tgt_texts_raw': texts,
            'texts_inp': texts_inp.long(),
            'texts_res': texts_res.long(),
            'text_masks': text_masks.long(),
        }


    def visualize_item(self, index = None, figsize=(15,15)):
        """
        Visualize an image with its captions by index
        """

        if index is None:
            index = random.randint(0,len(self.coco.imgs)-1)
        image_path = self.load_image(index)
        image_name = os.path.basename(image_path)
        image, _ = self.load_augment(image_path)

        texts = self.load_annotations(index, return_all=True)
        
        normalize = False
        if self.transforms is not None:
            for x in self.transforms.transforms[1]:
                if isinstance(x, A.Normalize):
                    normalize = True
                    denormalize = Denormalize(mean=x.mean, std=x.std)

        # Denormalize and reverse-tensorize
        if normalize:
            image = denormalize(img = image)

        self.visualize(image, texts, figsize = figsize, img_name= image_name)

    def visualize(self, img, texts, figsize=(15,15), img_name=None):
        """
        Visualize an image with its captions
        """

        text = []
        for i, t in enumerate(texts):
            text.append(f"{i+1}. {t}")
        text = "\n".join(text)
        fig = draw_image_caption(img, text, figsize=figsize)

        if img_name is not None:
            plt.title(img_name)
        plt.show()

    def count_dict(self, types = 1):
        """
        Count text length frequencies
        """
        cnt_dict = {}
        if types == 1: # Text length Frequencies
            for image_id in range(len(self.image_ids)):
                texts = self.load_annotations(image_id, return_all=True)
                for text in texts:
                    text_length = len(text)
                    if text_length not in cnt_dict.keys():
                        cnt_dict[text_length] = 0
                    cnt_dict[text_length] += 1
        
        return cnt_dict

    def plot(self, figsize = (8,8), types = ["length"]):
        """
        Plot distribution
        """
        ax = plt.figure(figsize = figsize)
        num_plots = len(types)
        plot_idx = 1

        if "length" in types:
            ax.add_subplot(num_plots, 1, plot_idx)
            plot_idx +=1
            cnt_dict = self.count_dict(types = 1)
            plt.title("Total texts: "+ str(sum(list(cnt_dict.values()))))
            bar1 = plt.bar(list(cnt_dict.keys()), list(cnt_dict.values()), color=[np.random.rand(3,) for i in range(len(cnt_dict.keys()))])
            for rect in bar1:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
        
        plt.show()

    def __str__(self): 
        s1 = "Number of images: " + str(len(self.image_ids)) + '\n'
        s2 = "Number of texts: " + str(len(self.coco.getAnnIds())) + '\n'
        return s1 + s2