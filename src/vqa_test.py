
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import collections
from pathlib import Path
from packaging import version
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import shutil
from pprint import pprint

from param import parse_args

from vqa_data import get_loader
from utils import load_state_dict, LossMeter, set_global_logging_level
import dist_utils
#import wandb
import logging
# set_global_logging_level(logging.ERROR, ["transformers"])

proj_dir = Path(__file__).resolve().parent.parent


_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from trainer_base import TrainerBase

class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

        from vqa_model import VLT5VQA, VLBartVQA

        model_kwargs = {}
        if 't5' in args.backbone:
            model_class = VLT5VQA
        elif 'bart' in args.backbone:
            model_class = VLBartVQA

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        if 'bart' in self.args.tokenizer:
            num_added_toks = 0
            if config.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

                config.default_obj_order_ids = self.tokenizer.convert_tokens_to_ids([f'<vis_extra_id_{i}>' for i in range(100)])

        self.model = self.create_model(model_class, config, **model_kwargs)

        if 't5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        elif 'bart' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.model.model.shared.num_embeddings + num_added_toks)

        self.model.tokenizer = self.tokenizer

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)

        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

    def predict(self, loader, dump_path=None):

        self.model.eval()
        with torch.no_grad():
            quesid2ans = {}

            pbar = tqdm(total=len(loader), ncols=120, desc="Prediction")
            for i, batch in enumerate(loader):

                # todo: add pgd attack here
                results = self.model.test_step(batch)

                pred_ans = results['pred_ans']
                ques_ids = batch['question_ids']

                for qid, ans in zip(ques_ids, pred_ans):
                    quesid2ans[qid] = ans

                pbar.update(1)

            pbar.close()

        qid2ans_list = dist_utils.all_gather(quesid2ans)
        if self.verbose:
            quesid2ans = {}
            for qid2ans in qid2ans_list:
                for k, v in qid2ans.items():
                    quesid2ans[k] = v

            if dump_path is not None:
                evaluator = loader.evaluator
                evaluator.dump_result(quesid2ans, dump_path)

        evaluator = self.test_loader.evaluator

        acc_dict_all = evaluator.evaluate_raw(quesid2ans)
        print("test"+acc_dict_all)

        return quesid2ans

    def predict1(self, loader, dump_path=None, epsilon=0.05, alpha=0.01, num_iter=5):

        self.model.eval()
        with torch.no_grad():
            quesid2ans = {}

            pbar = tqdm(total=len(loader), ncols=120, desc="Prediction")
            for i, batch in enumerate(loader):

                # PGD attack
                if epsilon > 0:
                    # Copy input data and create perturbation
                    X = batch['input_ids'].clone().detach().requires_grad_(True)
                    delta = torch.zeros_like(X)

                    for t in range(num_iter):
                        # Forward pass
                        outputs = self.model(X)
                        loss = outputs['loss']

                        # Calculate gradients
                        loss.backward()

                        # Create perturbation
                        delta_t = alpha * X.grad.detach().sign()
                        delta = torch.clamp(delta + delta_t, -epsilon, epsilon)

                        # Add perturbation to input data
                        X = torch.clamp(batch['input_ids'] + delta, 0, 1).detach().requires_grad_(True)

                    batch['input_ids'] = X

                # Perform regular prediction
                results = self.model.test_step(batch)

                pred_ans = results['pred_ans']
                ques_ids = batch['question_ids']

                for qid, ans in zip(ques_ids, pred_ans):
                    quesid2ans[qid] = ans

                pbar.update(1)

            pbar.close()

        qid2ans_list = dist_utils.all_gather(quesid2ans)
        if self.verbose:
            quesid2ans = {}
            for qid2ans in qid2ans_list:
                for k, v in qid2ans.items():
                    quesid2ans[k] = v

            if dump_path is not None:
                evaluator = loader.evaluator
                evaluator.dump_result(quesid2ans, dump_path)

        evaluator = self.test_loader.evaluator

        acc_dict_all = evaluator.evaluate_raw(quesid2ans)
        print("test" + acc_dict_all)

        return quesid2ans
    def evaluate(self, loader, dump_path=None):
        quesid2ans = self.predict(loader, dump_path)

        if self.verbose:
            evaluator = loader.evaluator
            acc_dict = evaluator.evaluate_raw(quesid2ans)
            # topk_score = evaluator.evaluate(quesid2ans)
            # acc_dict['topk_score'] = topk_score

            return acc_dict

def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    print(f'Building test loader at GPU {gpu}')
    test_loader = get_loader(
        args,
        split=args.test, mode='val', batch_size=args.valid_batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=4,
        topk=args.valid_topk,
    )

    trainer = Trainer(args, None, None, test_loader, train=False)

    qid_ans = trainer.predict(test_loader)
    qid_new = {}
    for key, val in qid_ans.items():
        qid_new[str(key)] = str(val)
    with open(args.output + 'predictions.json', 'w') as f:
        json.dump(qid_new, f)


if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    if args.local_rank in [0, -1]:
        print(args)

        comments = []
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)
        elif args.load_lxmert_qa is not None:
            ckpt_str = "_".join(args.load_lxmert_qa.split('/')[-3:])
            comments.append(ckpt_str)
        if args.comment != '':
            comments.append(args.comment)
        comment = '_'.join(comments)

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')

        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)

    else:
        main_worker(0, args)
