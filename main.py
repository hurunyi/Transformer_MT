import torch
from train.trainer import Trainer
from train.generator import build_generator
from data.dictionary import Dictionary
from data.dataloader import get_my_dataloader
from model.transformer.transformer_base import TransformerModelBase
from train.criterion import LabelSmoothedCrossEntropyCriterion
from argparse import ArgumentParser
from transformers import AdamW, get_linear_schedule_with_warmup

src_dict_fp = "/data/hurunyi/my_wmt14/dict.de.txt"
tgt_dict_fp = "/data/hurunyi/my_wmt14/dict.en.txt"
train_src_fp = "/data/hurunyi/my_wmt14/train.de"
train_tgt_fp = "/data/hurunyi/my_wmt14/train.en"
valid_src_fp = "/data/hurunyi/my_wmt14/valid.de"
valid_tgt_fp = "/data/hurunyi/my_wmt14/valid.en"
test_src_fp = "/data/hurunyi/my_wmt14/test.de"
test_tgt_fp = "/data/hurunyi/my_wmt14/test.en"


def main(args):
    # data
    print("Loading src and tgt dict...")
    src_dict = Dictionary.load(src_dict_fp)
    tgt_dict = Dictionary.load(tgt_dict_fp)

    train_dataloader = None
    valid_dataloader = None
    test_dataloader = None
    warmup_steps = 0
    t_total = 0
    best_bleu = 0
    begin_epoch = 0

    # data
    if args.do_test:
        print("Loading test data...")
        test_dataloader = get_my_dataloader(test_src_fp, test_tgt_fp, src_dict, tgt_dict, args.test_batch_size)
    else:
        print("Loading train and valid data...")
        train_dataloader = get_my_dataloader(train_src_fp, train_tgt_fp, src_dict, tgt_dict, args.train_batch_size)
        valid_dataloader = get_my_dataloader(valid_src_fp, valid_tgt_fp, src_dict, tgt_dict, args.valid_batch_size)
        t_total = int(len(train_dataloader) * args.epochs)
        warmup_steps = int(t_total * args.warmup_proportion)

    # model
    model = TransformerModelBase.build_model(args, src_dict, tgt_dict)
    generator = build_generator(model, args)
    criterion = LabelSmoothedCrossEntropyCriterion(args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    if args.load_from_checkpoint or args.do_test:
        print("Loading checkpoint...")
        checkpoint_state = torch.load("best_checkpoint.pt")
        begin_epoch = checkpoint_state["epoch"]
        best_bleu = checkpoint_state["best_bleu"]
        model.load_state_dict(checkpoint_state["model"])
        generator.load_state_dict(checkpoint_state["generator"])
        criterion.load_state_dict(checkpoint_state["criterion"])
        optimizer.load_state_dict(checkpoint_state["optimizer"])
        lr_scheduler.load_state_dict(checkpoint_state["lr_scheduler"])

    trainer = Trainer(args, model, generator, criterion, optimizer, lr_scheduler)

    if args.do_test:
        trainer.test(test_dataloader)
    else:
        trainer.train(train_dataloader, valid_dataloader, best_bleu, begin_epoch, seed=42)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--valid_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=512)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_proportion", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=7e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--max_len_a", type=float, default=1.2)
    parser.add_argument("--max_len_b", type=int, default=10)

    parser.add_argument("--load_from_checkpoint", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    args = parser.parse_args()

    main(args)
