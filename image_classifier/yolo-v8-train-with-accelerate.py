import argparse
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger, logging
import json
import evaluate
import math
import os
import torch
from tqdm import tqdm
import sys
from transformers import SchedulerType, get_scheduler
from types import SimpleNamespace
from torch.utils.data import DataLoader
from pathlib import Path
from ultralytics.data import ClassificationDataset
from ultralytics.nn.tasks import torch_safe_load, ClassificationModel


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Transformers model on an image classification dataset")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="/mnt/data2/image_classifier_data",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="/home/yuzhong/data1/models/yolo/yolov8m-cls.pt",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=32, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--every_log_steps",
        type=int,
        default=10,
        help="Number of steps to log loss.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="epoch",
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument("--seed", type=int, default=24, help="A seed for reproducible training.")

    args = parser.parse_args()

    # Sanity checks
    if not args.dataset_name:
        raise ValueError("no dataset name folder.")

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    return args


def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    # 设置日志文件名
    log_file_name = f'{file_name}.log'
    # 创建基础配置
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(log_file_name, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = get_logger(__name__)
    logger.info(accelerator.state, main_process_only=False)
    logger.info(f"开始执行训练,训练参数:{args}")
    logger.info(accelerator.state)
    logger.info(accelerator.state, main_process_only=False)
    if args.seed is not None:
        set_seed(args.seed)

    # 加载数据
    # 数据增强的一些配置
    data = {
        "imgsz": 224,
        "erasing": 0.4,
        "auto_augment": "randaugment",
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "bgr": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "copy_paste_mode": "flip",
        "crop_fraction": 1.0,
        "fraction": 1.0,
        "cache": False
    }
    nc = 0
    with accelerator.main_process_first():
        augemnt_args = SimpleNamespace(**data)
        data_dir = Path(args.dataset_name)
        train_set = data_dir / "train"
        test_set = data_dir / "test" if (data_dir / "test").exists() else None
        nc = len([x for x in (data_dir / "train").glob("*") if x.is_dir()])  # number of classes
        names = [x.name for x in (data_dir / "train").iterdir() if x.is_dir()]  # class names list
        names = dict(enumerate(sorted(names)))
        logging.info(f"class num: {nc}, names: {names}")
        train_dataset = ClassificationDataset(train_set, args=augemnt_args, augment=True, prefix="train")
        test_dataset = ClassificationDataset(test_set, args=augemnt_args, augment=False, prefix="test")
    # # DataLoaders creation:
    # def collate_fn(examples):
    #     pixel_values = torch.stack([example["img"] for example in examples])
    #     labels = torch.tensor([example[args.label_column_name] for example in examples])
    #     return {"pixel_values": pixel_values, "labels": labels}

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size, num_workers=32, prefetch_factor=2, pin_memory=True, drop_last=True
    )
    eval_dataloader = DataLoader(
        test_dataset, batch_size=args.per_device_train_batch_size, num_workers=32, prefetch_factor=2, pin_memory=True, drop_last=True
    )

    # 模型加载
    # 加载yolo 模型
    def load_yolo_model(path, nc):
        ckpt, w = torch_safe_load(path)
        model = ckpt["model"]
        ClassificationModel.reshape_outputs(model, nc)
        for p in model.parameters():
            p.requires_grad = True  # for training
        model = model.float()
        return model
    model = load_yolo_model("/mnt/data1/models/yolo/yolov8m-cls.pt", nc)

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    logger.info(f"pre preprare train_dataloader len: {len(train_dataloader)}, pre preprare eval_dataloader len: {len(eval_dataloader)}")
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )
    logger.info(f"pre preprare max_train_steps: {args.max_train_steps}, num_warmup_steps: {args.num_warmup_steps}, accelerator.num_processes:"
                f" {accelerator.num_processes}")
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    logger.info(f"Total samples: {len(train_dataset)}")
    logger.info(f"GPU num: {accelerator.num_processes}, Samples per GPU: {len(train_dataset) // accelerator.num_processes}")
    logger.info(f"train_dataloader len: {len(train_dataloader)}, eval_dataloader len: {len(eval_dataloader)}")
    # 分布式情况，dataloader会平均分配到不同的gpu上,所以每个epoch的数量会产生变化
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)
    metric = evaluate.load("accuracy")
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(args.output_dir) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]
        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                loss, loss_items = model(batch)
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss_items.float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if step % args.every_log_steps == 0 and accelerator.is_main_process:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    logging.info(f"Epoch: {epoch}, Step: {step}, Loss: {loss_items}, "
                                 f"total_loss/step: {total_loss.item() / (step * args.per_device_train_batch_size + 0.001)}, "
                                 f"Current learning rate: {current_lr}")

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if accelerator.is_main_process and isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    os.makedirs(output_dir, exist_ok=True)
                    accelerator.save_state(output_dir)
            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        logging.info("***** Evaluating *****")
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(batch['img'])
            predictions = outputs.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["cls"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")

        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if accelerator.is_main_process and args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
                os.makedirs(output_dir, exist_ok=True)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)


if __name__ == "__main__":
    main()
