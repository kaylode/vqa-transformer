import sys
sys.path.insert(0, './metrics/vqaeval')
from vqa import VQA

import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from augmentations.transforms import get_resize_augmentation, get_augmentation, Denormalize

from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

from .utils import create_masks, make_feature_batch
from utils.utils import draw_image_caption

class CocoDataset(Dataset):
    """
    Coco dataset
    """
    def __init__(self, 
            root_dir, ann_path, 
            question_path, tokenizer, 
            class_path = None,
            image_size=[224,224], 
            keep_ratio=False,
            type='train'):

        self.patch_size = 16
        self.root_dir = root_dir
        self.ann_path = ann_path
        self.question_path = question_path
        self.image_size = image_size
        self.type = type

        self.tokenizer = tokenizer
        self.transforms = A.Compose([
            get_resize_augmentation(image_size, keep_ratio=keep_ratio),
            get_augmentation(_type=type)
        ])

        self.coco = VQA(
            annotation_file=self.ann_path, 
            question_file=self.question_path,
            answer_file=class_path)
        
        self.image_ids = self.coco.getImgIds()

        if class_path is None:
            self.mapping_classes()
        else:
            self.load_mapping(class_path)

    def __len__(self):
        return len(self.image_ids)

    def mapping_classes(self):
        vocab = {}
        classes_idx = {}
        idx_classes = {}
        
        image_ids = self.coco.getImgIds()
        for img_id in image_ids:
            ann_ids = self.coco.getQuesIds(imgIds=img_id)
            anns = self.coco.loadQA(ann_ids)
            for ann in anns:
                ans = ann['answers']
                for an in ans:
                    if an['answer'] not in vocab.keys():
                        vocab[an['answer']] = 0
                    vocab[an['answer']] += 1

        sorted_vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1], reverse=True)}
        sorted_vocab =  {k: v for i, (k, v) in enumerate(sorted_vocab.items()) if i < 3129}

        for i, k in enumerate(sorted_vocab.keys()):
            classes_idx[k] = i
            idx_classes[i] = k

        self.num_classes = len(sorted_vocab.keys())
        self.classes_idx = classes_idx
        self.idx_classes = idx_classes

    def load_mapping(self, path):
        with open(path, 'r') as f:
            lines = f.read()
            lines = lines.splitlines()
      
        classes_idx = {}
        idx_classes = {}
        for i, k in enumerate(lines):
            classes_idx[k] = i
            idx_classes[i] = k

        self.classes_idx = classes_idx
        self.idx_classes = idx_classes
        self.num_classes = len(self.classes_idx)

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        return image_path

    def load_annotations(self, image_index, return_all=False):
        # get ground truth annotations

        annotations_ids = self.coco.getQuesIds(imgIds=self.image_ids[image_index])  

        if not return_all:
            ann_id = random.choice(annotations_ids)
            anns = self.coco.loadQA(ann_id)[0]
            quesId = anns['question_id']
            ques = self.coco.qqa[quesId]['question']

            anns = random.choice(anns['answers'])
            anns = anns['answer']
          
        else:
            annss = self.coco.loadQA(annotations_ids)

            anns = []
            ques = []
            quesIds = []
            for ann in annss:
                quesId = ann['question_id']
                question = self.coco.qqa[quesId]['question']
                an = []
                for answer in ann['answers']:
                    an.append(answer['answer'])
                anns.append(an)
                ques.append(question)
                quesIds.append(quesId)

        return anns, ques, quesId

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = self.load_image(index)
        ans, ques, quesId = self.load_annotations(index)
        label = self.classes_idx[ans]

        return {
            'image_id': image_id,
            'question_id': quesId,
            'image_path': image_path,
            'text': ques,
            'label': torch.LongTensor([label])
        }

    def load_augment(self, image_path):
        ori_img = cv2.imread(image_path)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        image = ori_img.astype(np.float32)
        image /= 255.0
        image = self.transforms(image=image)['image']
        return image, ori_img

    def collate_fn(self, batch):
        
        image_paths = [s['image_path'] for s in batch]
        image_ids = [s['image_id'] for s in batch]
        question_ids = [s['question_id'] for s in batch]
        labels = torch.stack([s['label'] for s in batch])
        
        image_names = []
        ori_imgs = []
        for image_path in image_paths:
            image_names.append(os.path.basename(image_path))

        imgs = []
        for image_path in image_paths:
            image, ori_img = self.load_augment(image_path)
            imgs.append(image)
            ori_imgs.append(ori_img)
        feats = torch.stack(imgs)
        mask_shapes = int((self.image_size[0] / self.patch_size) **2)
        image_masks = torch.ones((feats.shape[0], mask_shapes))

        texts = [s['text'] for s in batch]
        
        tokens = self.tokenizer(texts, truncation=True)
        tokens = [np.array(i) for i in tokens['input_ids']]

        texts_inp = make_feature_batch(
            tokens, pad_token=self.tokenizer.pad_token_id)
        

        text_masks = create_masks(
            texts_inp,
            pad_token=self.tokenizer.pad_token_id, 
            is_tgt_masking=True)
        
        texts_inp = texts_inp.squeeze(-1)

        return {
            'image_ids': image_ids,
            'question_ids': question_ids,
            'targets': labels.squeeze().long(),
            'image_names': image_names,
            'ori_imgs': ori_imgs,
            'image_patches': feats,
            'image_masks': image_masks.long(),
            'tgt_texts_raw': texts,
            'texts_inp': texts_inp.long(),
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

        ans, ques, quesId = self.load_annotations(index, return_all=True)
        
        ques_dict = [{'quesid': k, 'text': v} for k, v in zip(quesId, ques)]

        normalize = False
        if self.transforms is not None:
            for x in self.transforms.transforms[1]:
                if isinstance(x, A.Normalize):
                    normalize = True
                    denormalize = Denormalize(mean=x.mean, std=x.std)

        # Denormalize and reverse-tensorize
        if normalize:
            image = denormalize(img = image)

        self.visualize(image, ans, ques_dict, figsize = figsize, img_name= image_name)

    def visualize(self, img, answers, questions, figsize=(15,15), img_name=None):
        """
        Visualize an image with its captions
        """

        text = []

        for question, answer in zip(questions, answers):
            text2 = []
            for i, ans in enumerate(answer):
                text2.append(f"{ans}")
            text2 = "-".join(text2)
            text2 = f"{question['text']} " + text2
            text.append(text2)
        text = "\n".join(text)
        fig = draw_image_caption(img, text, figsize=figsize)

        if img_name is not None:
            plt.title(img_name)
        plt.show()

    def __str__(self): 
        s1 = "Number of images: " + str(len(self.image_ids)) + '\n'
        return s1

class NumpyFeatureDataset(Dataset):
    """
    Coco dataset
    """
    def __init__(self, 
        root_dir, ann_path, 
        tokenizer, npy_dir,
        question_path,  class_path = None):

        self.root_dir = root_dir
        self.question_path = question_path
        self.class_path = class_path
        self.ann_path = ann_path
        self.npy_dir = npy_dir
        self.tokenizer = tokenizer
        self.coco = VQA(
            annotation_file=self.ann_path, 
            question_file=self.question_path,
            answer_file=class_path)
        self.image_ids = self.coco.getImgIds()

        if class_path is None:
            self.mapping_classes()
        else:
            self.load_mapping(class_path)

    def get_feature_dim(self):
        return 2048 # bottom up attention features

    def __len__(self):
        return len(self.image_ids)

    def mapping_classes(self):
        vocab = {}
        classes_idx = {}
        idx_classes = {}
        
        image_ids = self.coco.getImgIds()
        for img_id in image_ids:
            ann_ids = self.coco.getQuesIds(imgIds=img_id)
            anns = self.coco.loadQA(ann_ids)
            for ann in anns:
                ans = ann['answers']
                for an in ans:
                    if an['answer'] not in vocab.keys():
                        vocab[an['answer']] = 0
                    vocab[an['answer']] += 1

        sorted_vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1], reverse=True)}
        sorted_vocab =  {k: v for i, (k, v) in enumerate(sorted_vocab.items()) if i < 3129}

        for i, k in enumerate(sorted_vocab.keys()):
            classes_idx[k] = i
            idx_classes[i] = k

        self.num_classes = len(sorted_vocab.keys())
        self.classes_idx = classes_idx
        self.idx_classes = idx_classes

    def load_mapping(self, path):
        with open(path, 'r') as f:
            lines = f.read()
            lines = lines.splitlines()
      
        classes_idx = {}
        idx_classes = {}
        for i, k in enumerate(lines):
            classes_idx[k] = i
            idx_classes[i] = k

        self.classes_idx = classes_idx
        self.idx_classes = idx_classes
        self.num_classes = len(self.classes_idx)

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        return image_path

    def load_numpy(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        npy_path = os.path.join(self.npy_dir+'_att', image_info['id']+'.npz')
        npy_loc_path = os.path.join(self.npy_dir+'_box', image_info['id']+'.npz')
        return npy_path, npy_loc_path

    def load_annotations(self, image_index, return_all=False):
        # get ground truth annotations

        annotations_ids = self.coco.getQuesIds(imgIds=self.image_ids[image_index])  

        if not return_all:
            ann_id = random.choice(annotations_ids)
            anns = self.coco.loadQA(ann_id)[0]
            quesId = anns['question_id']
            ques = self.coco.qqa[quesId]['question']

            anns = random.choice(anns['answers'])
            anns = anns['answer']
          
        else:
            annss = self.coco.loadQA(annotations_ids)

            anns = []
            ques = []
            for ann in annss:
                quesId = ann['question_id']
                question = self.coco.qqa[quesId]['question']
                an = []
                for answer in ann['answers']:
                    an.append(answer['answer'])
                anns.append(an)
                ques.append(question)

        return anns, ques

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = self.load_image(index)
        ans, ques = self.load_annotations(index)
        npy_path, npy_loc_path = self.load_numpy(index)
        label = self.classes_idx[ans]

        return {
            'image_id': image_id,
            'image_path': image_path,
            'npy_path': npy_path,
            'npy_loc_path': npy_loc_path,
            'text': ques,
            'label': torch.LongTensor([label])
        }

    def load_augment(self, image_path):
        ori_img = cv2.imread(image_path)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        image = ori_img.astype(np.float32)
        image /= 255.0
        image = self.transforms(image=image)['image']
        return image, ori_img

    def collate_fn(self, batch):
        
        image_paths = [s['image_path'] for s in batch]
        npy_paths = [s['npy_path'] for s in batch]
        npy_loc_paths = [s['npy_loc_path'] for s in batch]
        image_ids = [s['image_id'] for s in batch]
        labels = torch.stack([s['label'] for s in batch])

        image_names = []
        ori_imgs = []
        for image_path in image_paths:
            image_names.append(os.path.basename(image_path))

        for image_path in image_paths:
            ori_img = cv2.imread(image_path)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            ori_imgs.append(ori_img)
        
        npy_feats = []
        npy_loc_feats = []
        for npy_path, npy_loc_path in zip(npy_paths, npy_loc_paths):
            npy_feat = np.load(npy_path, mmap_mode='r')['feat']
            npy_loc_feat = np.load(npy_loc_path, mmap_mode='r')['feat']
            npy_feats.append(npy_feat)
            npy_loc_feats.append(npy_loc_feat)

        npy_feats = np.stack(npy_feats, axis=0)
        npy_loc_feats = np.stack(npy_loc_feats, axis=0)

        feats = torch.from_numpy(npy_feats).float()
        loc_feats = torch.from_numpy(npy_loc_feats).float()

        image_masks = torch.ones(feats.shape[:2])

        texts = [s['text'] for s in batch]
        
        tokens = self.tokenizer(texts, truncation=True)
        tokens = [np.array(i) for i in tokens['input_ids']]

        texts_inp = make_feature_batch(
            tokens, pad_token=self.tokenizer.pad_token_id)
        
        text_masks = create_masks(
            texts_inp,
            pad_token=self.tokenizer.pad_token_id, 
            is_tgt_masking=True)
        
        texts_inp = texts_inp.squeeze(-1)

        return {
            'image_ids': image_ids,
            'image_names': image_names,
            'ori_imgs': ori_imgs,
            'feats': feats,
            'loc_feats': loc_feats,
            'targets': labels.squeeze().long(),
            'image_masks': image_masks.long(),
            'tgt_texts_raw': texts,
            'texts_inp': texts_inp.long(),
            'text_masks': text_masks.long(),
        }

    def __str__(self): 
        s1 = "Number of images: " + str(len(self.image_ids)) + '\n'
        return s1
