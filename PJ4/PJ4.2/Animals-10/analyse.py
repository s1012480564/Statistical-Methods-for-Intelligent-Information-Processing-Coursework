import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os


def compute_and_save_confusion_matrix(labels: np.ndarray, preds: np.ndarray):
    category_names = os.listdir("data/test")
    cm = confusion_matrix(labels, preds)
    cm = cm / np.sum(cm, axis=-1).reshape(-1, 1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=category_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("glip")
    plt.savefig("outputs/glip-confusion_matrix.png", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    labels = np.load("outputs/animals_10-test_labels.npy")
    preds = np.load("outputs/glip/glip-animals_10_aug-250104-1413-test_preds.npy")
    compute_and_save_confusion_matrix(labels, preds)
