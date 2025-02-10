import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets import Dataset
from sklearn import metrics
from typing import Dict
from config_utils import Args
from time import strftime, localtime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os


def evaluate(model: nn.Module, test_dataset: Dataset, args: Args, save_test_preds: bool = True) -> Dict[str, float]:
    test_dataset = test_dataset.with_format("torch", device=args.device)
    data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)
    labels_all, preds_all = None, None
    with torch.no_grad():
        for i_batch, batch in enumerate(data_loader):
            logits = model(**batch)["logits"]
            labels = batch["labels"]
            preds = logits.argmax(axis=-1)
            labels_all = torch.cat([labels_all, labels]) if labels_all is not None else labels
            preds_all = torch.cat([preds_all, preds]) if preds_all is not None else preds
    labels_all = labels_all.cpu().numpy()
    preds_all = preds_all.cpu().numpy()
    if save_test_preds:
        np.save(
            f"{args.output_dir}/{args.model_name}-{args.dataset_name}-{strftime("%y%m%d-%H%M", localtime())}-test_preds.npy",
            preds_all)
    acc = round(metrics.accuracy_score(labels_all, preds_all), 4)
    macro_f1 = round(metrics.f1_score(labels_all, preds_all, labels=range(args.num_classes), average="macro"), 4)
    return {"test_accuracy": acc, "test_macro_f1": macro_f1}


# def test_unit():
#     from models import VGG
#     from torch.nn import CrossEntropyLoss
#     from data_utils import preprocess, get_transform
#     from datasets import load_dataset
#     from functools import partial
#
#     args = Args(device=0, batch_size=16, num_classes=6, dropout=0.1, criterion=CrossEntropyLoss(),
#                 pretrained_path="../../pretrained/vgg19-bn/vgg19_bn-c79401a0.pth")
#
#     dataset = load_dataset("imagefolder", data_dir="data_small")
#     transform = get_transform(args)
#     dataset = dataset.map(partial(preprocess, transform=transform), batched=True, batch_size=None,
#                           remove_columns=["image", "label"])
#     test_dataset = dataset["test"]
#
#     model = VGG(args).to(args.device)
#
#     print(evaluate(model, test_dataset, args))


def compute_and_save_confusion_matrix(labels: np.ndarray, preds: np.ndarray):
    category_names = os.listdir("data/test")
    cm = confusion_matrix(labels, preds)
    cm = cm / np.sum(cm, axis=-1).reshape(-1, 1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=category_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("vgg19_bn")
    plt.savefig("outputs/vgg19_bn-confusion_matrix.png", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    labels = np.load("outputs/animals_10-test_labels.npy")
    preds = np.load("outputs/vgg19_bn/vgg19_bn-animals_10-241217-2303-test_preds.npy")
    compute_and_save_confusion_matrix(labels, preds)
