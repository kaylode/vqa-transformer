from .cocoset import CocoDataset, NumpyFeatureDataset, ValNumpyFeatureDataset
from torch.utils.data import DataLoader
from torchtext.legacy.data import BucketIterator


class EqualLengthTextLoader(BucketIterator):
    """
    Use BucketIterator to make texts of same length into batch
    """
    def __init__(self, 
                batch_size,
                root_dir,
                ann_path,
                question_path,
                class_path,
                tokenizer,
                image_size,
                keep_ratio,
                device,
                type,
                **kwargs):
       
        self.dataset = CocoDataset(
                root_dir=root_dir, ann_path=ann_path, 
                question_path=question_path, class_path=class_path,
                tokenizer=tokenizer, image_size=image_size, 
                keep_ratio=keep_ratio, type=type)
        

        self.tokenizer = self.dataset.tokenizer
        self.collate_fn = self.dataset.collate_fn
        
        super(EqualLengthTextLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            device=device, 
            sort_key=lambda x: len(x['text']),
            repeat=True, # Repeat the iterator for multiple epochs.
            sort=False,  # Sort all examples in data using `sort_key`.
            shuffle=True, # Shuffle data on each epoch run.
            sort_within_batch=True) # Use `sort_key` to sort examples in each batch.


class RawTextLoader(DataLoader):
    """
    Use DataLoader to make texts into batch
    """
    def __init__(self, 
                batch_size,
                root_dir,
                ann_path,
                question_path,
                class_path,
                tokenizer,
                image_size,
                keep_ratio,
                type,
                **kwargs):
       
        self.dataset = CocoDataset(
                root_dir=root_dir, ann_path=ann_path, 
                question_path=question_path, class_path=class_path,
                tokenizer=tokenizer, image_size=image_size, 
                keep_ratio=keep_ratio,type=type)

        self.tokenizer = self.dataset.tokenizer
        self.collate_fn = self.dataset.collate_fn
        
        super(RawTextLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            collate_fn=self.collate_fn)

class NumpyFeatureLoader(BucketIterator):
    """
    Use BucketIterator to make texts of same length into batch
    """
    def __init__(self, 
                batch_size,
                root_dir,
                npy_dir, 
                ann_path, 
                question_path,
                class_path,
                tokenizer,
                device,
                **kwargs):
       
        self.dataset = NumpyFeatureDataset(
            root_dir, ann_path, tokenizer, npy_dir, question_path, class_path)

        self.tokenizer = self.dataset.tokenizer
        self.collate_fn = self.dataset.collate_fn
        
        super(NumpyFeatureLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            device=device, 
            sort_key=lambda x: len(x['text']),
            repeat=True, # Repeat the iterator for multiple epochs.
            sort=False,  # Sort all examples in data using `sort_key`.
            shuffle=True, # Shuffle data on each epoch run.
            sort_within_batch=True) # Use `sort_key` to sort examples in each batch.

class RawNumpyFeatureLoader(DataLoader):
    """
    Use DataLoader to make texts into batch
    """
    def __init__(self, 
                batch_size,
                root_dir,
                npy_dir, 
                ann_path, 
                question_path,
                class_path,
                tokenizer,
                **kwargs):
       
        self.dataset = ValNumpyFeatureDataset(
            root_dir, ann_path, tokenizer, npy_dir, question_path, class_path)

        self.tokenizer = self.dataset.tokenizer
        self.collate_fn = self.dataset.collate_fn
        
        super(RawNumpyFeatureLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            collate_fn=self.collate_fn)
