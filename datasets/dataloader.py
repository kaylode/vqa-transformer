from .dataset import ImageTextSet
from torch.utils.data import DataLoader
from torchtext.legacy.data import BucketIterator


class EqualLengthTextLoader(BucketIterator):
    """
    Use BucketIterator to make texts of same length into batch
    """
    def __init__(self, 
                batch_size,
                input_path,
                csv_file,
                tokenizer,
                image_size,
                keep_ratio,
                device,
                **kwargs):
       
        self.dataset = ImageTextSet(
                input_path, csv_file, tokenizer, 
                image_size=image_size, keep_ratio=keep_ratio)

        self.stokenizer = self.dataset.tokenizer
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
                input_path,
                csv_file,
                tokenizer,
                image_size,
                keep_ratio,
                **kwargs):
       
        self.dataset = ImageTextSet(
                input_path, csv_file, tokenizer, 
                image_size=image_size, keep_ratio=keep_ratio)

        self.stokenizer = self.dataset.tokenizer
        self.collate_fn = self.dataset.collate_fn
        
        super(RawTextLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            collate_fn=self.collate_fn)