# coding=utf-8
import argparse
import os
import logging
import random
import numpy as np
from datetime import timedelta

import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    config = CONFIGS[args.model_type]
    num_classes = 2 if args.dataset == "ants_bees" else (10 if args.dataset == "cifar10" else 100)

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    if args.pretrained_dir:
        model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)

    # CORRECTED debugging code:
    print("Classifier weight norms:", model.head.weight.norm().item())
    print("First block attention query norms:",
          model.transformer.encoder.layer[0].attn.query.weight.norm().item())

    num_params = count_parameters(model)
    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    eval_losses = AverageMeter()
    model.eval()
    all_preds, all_label = [], []

    logger.info("***** Running Validation *****")
    for step, batch in enumerate(test_loader):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]
            eval_loss = torch.nn.CrossEntropyLoss()(logits, y)
            eval_losses.update(eval_loss.item())
            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
            all_label[0] = np.append(all_label[0], y.detach().cpu().numpy(), axis=0)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\nValidation Results")
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    if writer:
        writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


def train(args, model):
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    train_loader, test_loader = get_loader(args)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999))

    t_total = args.num_steps
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)

    model.zero_grad()
    losses = AverageMeter()
    global_step, best_acc = 0, 0

    while global_step < args.num_steps:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc=f"Training ({global_step}/{args.num_steps})",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)

        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch

            # Debug input
            print(f"Input range: {x.min().item():.4f} to {x.max().item():.4f}")
            print(f"Labels: {y.tolist()}")

            logits = model(x)[0]
            loss = torch.nn.CrossEntropyLoss()(logits, y)

            # Debug output
            print(f"Step {global_step}: Loss = {loss.item():.4f}")
            print(f"Logits sample: {logits[0].detach().cpu().numpy()}")

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None and 'head' in name:
                    print(f"{name} grad: {param.grad.norm().item():.6f}")
            losses.update(loss.item())

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if global_step % 10 == 0:
                    print(f"Gradient norm: {total_norm:.4f}")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy = valid(args, model, writer, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                    model.train()

                if global_step >= args.num_steps:
                    break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "ants_bees"], default="ants_bees")
    parser.add_argument("--model_type",
                        choices=["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16")
    parser.add_argument("--pretrained_dir", type=str, default="pretrained/ViT-B_16.npz",
                        help="Path to pretrained model")
    parser.add_argument("--output_dir", default="output", type=str, help="Output directory")
    parser.add_argument("--data_dir", default="./Data", type=str, help="Data directory")
    parser.add_argument("--img_size", default=224, type=int, help="Image size")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Training batch size")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Evaluation batch size")
    parser.add_argument("--eval_every", default=100, type=int, help="Evaluation frequency")
    parser.add_argument("--learning_rate", default=3e-3, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay")
    parser.add_argument("--num_steps", default=1000, type=int, help="Number of training steps")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--warmup_steps", default=100, type=int, help="Warmup steps")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    args = parser.parse_args()

    # Setup device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count() if args.device.type == "cuda" else 0

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s" %
                   (args.local_rank, args.device, args.n_gpu))

    set_seed(args)
    args, model = setup(args)
    train(args, model)


if __name__ == "__main__":
    main()