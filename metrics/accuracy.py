import os
import json
import torch
from tqdm import tqdm
from .metrictemplate import TemplateMetric
from .vqaeval import VQA, VQAEval

"""
GT format
annotation{
  "id": int, 
  "image_id": int, 
  "caption": str,
}
Result format
[{
    "question_id": int,
    "answer": str
}]
"""

def _eval(gt_ans_path, gt_ques_path, pred_path, class_path):

    vqa = VQA(
        annotation_file=gt_ans_path,
        question_file=gt_ques_path,
        answer_file=class_path
    )
    
    vqaRes = vqa.loadRes(pred_path, gt_ques_path)

    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes, n=3)   #n is precision of accuracy (number of places after decimal), default is 2

    vqaEval.evaluate()

    # create output dictionary
    stats = vqaEval.accuracy

    result_dict = {
        'accuracy': stats['overall']
    }
    # ['perQuestionType']
    # ['perAnswerType']
    return result_dict

class AccuracyMetric(TemplateMetric):
    def __init__(
            self,
            dataloader, 
            max_samples = None,
            decimals = 5):

        self.dataloader = dataloader
        self.max_samples = max_samples
        self.decimals = decimals
        self.filepath = f'results/results.json'
        self.gt_ans_filepath = self.dataloader.dataset.ann_path
        self.gt_ques_filepath = self.dataloader.dataset.question_path
        self.class_filepath = self.dataloader.dataset.class_path
        self.class_mapping = self.dataloader.dataset.idx_classes
        self.reset()

        if not os.path.exists('results'):
            os.mkdir('results')
            
    def reset(self):
        self.model = None

    def update(self, model):
        self.model = model
        self.model.eval()

    def compute(self):
        result_dict = []

        with torch.no_grad():
            if self.max_samples is not None:
                total_iter = min(len(self.dataloader)-1, int(self.max_samples/self.dataloader.batch_size))
            else:
                total_iter = len(self.dataloader)-1

            with tqdm(total=total_iter) as pbar:
                for idx, batch in enumerate(self.dataloader):
                    if idx > total_iter:
                        break
                    
                    questions = batch['question_ids']
                    preds = self.model.inference_step(batch)
                    
                    for question, pred in zip(questions, preds):
                            
                        result_dict.append({
                            "question_id": question,
                            "answer": self.class_mapping[int(pred)]
                        })
                                                    
                    pbar.update(1)

        if not len(result_dict):
            return False

        # write output
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        json.dump(result_dict, open(self.filepath, 'w'), indent=4)

        return True

    def value(self):
        self.compute()
        stats = _eval(
            self.gt_ans_filepath, 
            self.gt_ques_filepath,
            self.filepath,
            self.class_filepath)
        
        return stats

    def __str__(self):
        return f'{self.value()}'

    def __len__(self):
        return len(self.dataloader)