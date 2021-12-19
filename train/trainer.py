import warnings
import random
import os
import logging
import numpy as np

import torch
import math
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
import sacrebleu
from sacremoses import MosesDetokenizer

warnings.simplefilter('ignore')

logger = logging.getLogger(__name__)


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self, args, model, generator, criterion, optimizer, lr_scheduler):
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(device=self.device)
        self.tgt_dict = self.model.decoder.dictionary
        self.generator = generator.to(device=self.device)
        self.detok = MosesDetokenizer(lang='en')
        self.criterion = criterion.to(device=self.device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.max_epoch = args.epochs
        self.grad_clip = args.grad_clip

    def train_epoch(self, samples):
        """Do forward, backward and parameter update."""
        self.model.train()
        self.criterion.train()
        scaler = GradScaler()

        # forward and backward pass
        logging_outputs = []
        pbar = tqdm(enumerate(samples))
        pbar.set_description(desc="Train")
        losses = []
        for i, sample in pbar:  # delayed update loop
            sample = self.prepare_sample(sample)
            self.optimizer.zero_grad()
            # forward and backward
            with autocast():
                loss, logging_output = self.criterion(self.model, sample)
            scaler.scale(loss).backward()
            losses.append(scaler.scale(loss).item())
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            scaler.step(self.optimizer)
            self.lr_scheduler.step()
            scaler.update()

            logging_outputs.append(logging_output)
            if i % 50 == 0:
                pbar.set_postfix(loss=np.mean(losses))

        return logging_outputs

    @staticmethod
    def inference_step(generator, sample):
        with torch.no_grad():
            return generator.generate(sample)

    def inference(self, generator, sample, model):
        def strip_pad(tensor, pad):
            return tensor[tensor.ne(pad)]

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                '@@ ',
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.detok:
                s = self.detok.detokenize(tokens=s.split())
            return s

        gen_out = self.inference_step(generator, sample)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        return hyps, refs

    def valid_epoch(self, samples):
        """Do forward pass in evaluation mode."""
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()

            pbar = tqdm(enumerate(samples))
            pbar.set_description(desc="Valid")
            hyps_all, refs_all = [], []
            for i, sample in pbar:
                sample = self.prepare_sample(sample)
                hyps, refs = self.inference(self.generator, sample, self.model)
                hyps_all += hyps
                refs_all += refs

        return sacrebleu.corpus_bleu(hyps_all, [refs_all]).score

    def prepare_sample(self, sample):
        sample["net_input"]["src_tokens"] = \
            sample["net_input"]["src_tokens"].to(device=self.device)
        sample["net_input"]["prev_output_tokens"] = \
            sample["net_input"]["prev_output_tokens"].to(device=self.device)
        sample["target"] = \
            sample["target"].to(device=self.device)
        return sample

    def test(self, test_data):
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()

            pbar = tqdm(enumerate(test_data))
            pbar.set_description(desc="Test")
            hyps_all, refs_all = [], []
            for i, sample in pbar:
                sample = self.prepare_sample(sample)
                hyps, refs = self.inference(self.generator, sample, self.model)
                hyps_all += hyps
                refs_all += refs
            logger.info(sacrebleu.corpus_bleu(hyps_all, [refs_all]).score)
            with open("hyps_refs.txt", "w") as f:
                for hyp, ref in zip(hyps_all, refs_all):
                    f.write(f"hyp:{hyp}\nref:{ref}\n\n")

    def train(self, train_data, valid_data, best_bleu, begin_epoch, seed):
        seed_everything(seed)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

        max_epoch = self.max_epoch or math.inf
        for epoch_id in range(max_epoch):
            # train for one epoch
            print("\n")
            logger.info(f"Epoch: {epoch_id + 1 + begin_epoch}")
            _ = self.train_epoch(train_data)
            bleu = self.valid_epoch(valid_data)
            logger.info(f"BLEU: {bleu}, Old best BLEU: {best_bleu}")

            if bleu > best_bleu:
                best_bleu = bleu
                logger.info(f"New best BLEU: {best_bleu}!")
                torch.save(
                    {
                        "epoch": epoch_id + 1 + begin_epoch,
                        "best_bleu": best_bleu,
                        "model": self.model.state_dict(),
                        "generator": self.generator.state_dict(),
                        "criterion": self.criterion.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "lr_scheduler": self.lr_scheduler.state_dict(),
                    },
                    "best_checkpoint.pt"
                )
                logger.info("New best model saved!")
