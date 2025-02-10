import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, kaiming_normal_, xavier_uniform_, xavier_normal_
from sklearn import metrics
import os
import math
import argparse
from time import strftime, localtime
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, Trainer, EvalPrediction, EarlyStoppingCallback
from functools import partial
from typing import Dict
from config_utils import Args
from data_utils import preprocess, get_transform
from models import AlexNet, DenseNet, GoogLeNet, LeNet, ResNet, VGG, VisionTransformer
from evaluate import evaluate

model_classes = {
    "alexnet": AlexNet,
    "densenet201": DenseNet,
    "googlenet": GoogLeNet,
    "lenet5": LeNet,
    "resnet152": ResNet,
    "vgg19_bn": VGG,
    "vit_b_16": VisionTransformer,
}

pretrained_paths = {
    "alexnet": "../../pretrained/alexnet/alexnet-owt-7be5be79.pth",
    "densenet201": "../../pretrained/densenet201/densenet201-c1103571.pth",
    "googlenet": "../../pretrained/googlenet/googlenet-1378be20.pth",
    "lenet5": None,
    "resnet152": "../../pretrained/resnet152/resnet152-394f9c45.pth",
    "vgg19_bn": "../../pretrained/vgg19-bn/vgg19_bn-c79401a0.pth",
    "vit_b_16": "../../pretrained/vit-b-16/vit_b_16-c867db91.pth"
}

initializer_funcs = {
    'kaiming_uniform': kaiming_uniform_,
    'kaiming_normal': kaiming_normal_,
    'xavier_uniform': xavier_uniform_,
    'xavier_normal': xavier_normal_,
}


def _freeze_params(model: nn.Module, args: Args) -> None:
    match args.model_name:  # 低层次特征冻结，首次卷积出最大通道或者足够大通道开始视作高层次特征（拼接不算）
        case "alexnet":  # features[6], >=384
            for i in range(4):
                model.alexnet.features[i].requires_grad_(False)
        case "densenet201":  # transition2, >=256
            layer_names = ["conv0", "norm0", "denseblock1", "transition1", "denseblock2"]
            for attr in layer_names:
                getattr(model.densenet201.features, attr).requires_grad_(False)
        case "googlenet":  # inception4c, >=256
            layer_names = ["conv1", "conv2", "conv3", "inception3a", "inception3b", "inception4a", "inception4b"]
            for attr in layer_names:
                getattr(model.googlenet, attr).requires_grad_(False)
        case "resnet152":  # layer3, >=1024
            layer_names = ["conv1", "bn1", "layer1", "layer2"]
            for attr in layer_names:
                getattr(model.resnet152, attr).requires_grad_(False)
        case "vgg19_bn":  # features[14], >=256
            for i in range(12):
                model.vgg19_bn.features[i].requires_grad_(False)
        case "vit_b_16":  # vit 和上述不同，这里选择冻掉 patch_embed 和 encoder 至少 20% 的层数
            layer_names = ["encoder_layer_" + str(i) for i in range(3)]
            model.vit_b_16.conv_proj.requires_grad_(False)
            for attr in layer_names:
                getattr(model.vit_b_16.encoder.layers, attr).requires_grad_(False)
        case _:
            pass


def _init_params(model: nn.Module, args: Args) -> None:
    head = None  # 除 lenet，仅 head 初始化参数
    match args.model_name:
        case "alexnet" | "vgg19_bn":
            head = getattr(model, args.model_name).classifier[-1]
        case "densenet201":
            head = getattr(model, args.model_name).classifier
        case "googlenet" | "resnet152":
            head = getattr(model, args.model_name).fc
        case "vit_b_16":
            head = getattr(model, args.model_name).heads.head
    if args.model_name == "lenet5":
        for child in model.children():
            for p in child.parameters():
                if p.requires_grad:
                    if len(p.shape) > 1:
                        args.initializer(p)
                    else:
                        stdv = 1. / (p.shape[0] ** 0.5)
                        nn.init.uniform_(p, a=-stdv, b=stdv)
    else:
        for p in head.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    args.initializer(p)
                else:
                    stdv = 1. / (p.shape[0] ** 0.5)
                    nn.init.uniform_(p, a=-stdv, b=stdv)


def _compute_metrics(predictions: EvalPrediction, args: Args = None) -> Dict[str, float]:
    logits = predictions.predictions
    labels = predictions.label_ids
    preds = logits.argmax(axis=-1)
    acc = metrics.accuracy_score(labels, preds)
    macro_f1 = metrics.f1_score(labels, preds, labels=range(args.num_classes), average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}


def _save_log(model: nn.Module, dataset: Dataset, trainer: Trainer, args: Args, parser_args,
              save_test_labels: bool = False, save_test_preds: bool = True) -> None:
    n_trainable_params, n_nontrainable_params = 0, 0
    for p in model.parameters():
        if p.requires_grad:
            n_trainable_params += p.numel()
        else:
            n_nontrainable_params += p.numel()

    len_dataloader = math.ceil(len(dataset["train"]) / parser_args.batch_size)
    num_update_steps_per_epoch = len_dataloader // parser_args.gradient_accumulation_steps
    total_optimization_steps = math.ceil(parser_args.num_epochs * num_update_steps_per_epoch)

    log_file = f"logs/{args.model_name}-{args.dataset_name}-{strftime("%y%m%d-%H%M", localtime())}.log"
    with open(log_file, "w") as file:
        file.write(f"cuda_max_memory_allocated: {torch.cuda.max_memory_allocated()}\n")
        file.write(
            f"> num_trainable_parameters: {n_trainable_params}, num_nontrainable_parameters: {n_nontrainable_params}\n")
        file.write(f"> num_train_examples: {len(dataset["train"])}, num_test_examples: {len(dataset["test"])}\n")
        file.write(f"> total_optimization_steps: {total_optimization_steps}\n")
        file.write("> training arguments: \n")
        for arg in vars(parser_args):
            file.write(f">>> {arg}: {getattr(parser_args, arg)}\n")

        epoch = 0
        file.write('>' * 100 + '\n')
        file.write(f"epoch: {epoch}\n")
        for log in trainer.state.log_history:
            for key in log:
                log[key] = round(log[key], 4)

            if "eval_loss" in log:
                file.write(f"> {log}\n")
                epoch += 1
                if epoch < parser_args.num_epochs:
                    file.write('>' * 100 + '\n')
                    file.write(f"epoch: {epoch}\n")
            else:
                file.write(f"{log}\n")

        test_results = evaluate(model, dataset["test"], args, save_test_preds)
        file.write(f">>> {test_results}\n")

    if save_test_labels:
        np.save(f"{args.output_dir}/{args.dataset_name}-test_labels.npy", np.array(dataset["test"]["labels"]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="vgg19_bn", type=str)
    parser.add_argument("--dataset_name", default="animals_10", type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--l2reg", default=1e-5, type=float)
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--optimizer_name", default="adamw", type=str)
    parser.add_argument("--scheduler_type", default="constant", type=str)
    parser.add_argument("--initializer_name", default="kaiming_uniform", type=str)
    parser.add_argument("--warmup_ratio", default=0, type=float)
    parser.add_argument("--logging_steps", default=3, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--early_stopping_patience", default=5, type=int)
    parser.add_argument("--num_classes", default=6, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)

    parser_args = parser.parse_args()
    args = Args(**vars(parser_args))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    args.device = 0

    args.model_class = model_classes[args.model_name]
    args.initializer = initializer_funcs[args.initializer_name]
    args.criterion = nn.CrossEntropyLoss()
    args.pretrained_path = pretrained_paths[args.model_name]

    model = args.model_class(args)

    dataset = load_dataset("imagefolder", data_dir="data")
    transform = get_transform(args)
    dataset = dataset.map(partial(preprocess, transform=transform), batched=True, batch_size=None,
                          remove_columns=["image", "label"])
    train_dataset, test_dataset = dataset["train"], dataset["test"]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.l2reg,
        num_train_epochs=args.num_epochs,
        lr_scheduler_type=args.scheduler_type,
        warmup_ratio=args.warmup_ratio,
        log_level="info",
        logging_dir="logs",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=partial(_compute_metrics, args=args),
        callbacks=[EarlyStoppingCallback(args.early_stopping_patience)],
    )

    _freeze_params(model, args)
    _init_params(model, args)
    trainer.train()

    _save_log(model, dataset, trainer, args, parser_args)
