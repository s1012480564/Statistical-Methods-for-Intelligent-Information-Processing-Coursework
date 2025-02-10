import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# -------------------- 数据预处理与增强 --------------------
def preprocess_image(image):
    """基本图像预处理：调整大小、去噪、归一化"""
    image_resized = cv2.resize(image, (224, 224))
    image_blurred = cv2.GaussianBlur(image_resized, (5, 5), 0)
    return image_blurred / 255.0

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = preprocess_image(img)
            images.append(img)
            labels.append(label)
    return images, labels

def load_all_data(data_dirs):
    all_images = []
    all_labels = []
    for label, folder in enumerate(data_dirs):
        images, labels = load_images_from_folder(folder, label)
        all_images.extend(images)
        all_labels.extend(labels)
    return all_images, all_labels

# -------------------- 特征提取 --------------------
def extract_color_histogram(image):
    """提取颜色直方图（全局特征）"""
    image_hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image_hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_sift_features(image):
    """提取 SIFT 局部特征"""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute((image * 255).astype(np.uint8), None)
    return descriptors

def create_bow_histogram(descriptors, bow_kmeans, voc_size):
    """使用 BoW 模型将 SIFT 特征转换为特征向量"""
    if descriptors is None:
        return np.zeros(voc_size)
    words = bow_kmeans.predict(descriptors)
    histogram = np.zeros(voc_size)
    for word in words:
        histogram[word] += 1
    return histogram

def combine_features(color_features, bow_features):
    """早期融合，将颜色直方图和 BoW 特征连接"""
    return np.hstack((color_features, bow_features))

# -------------------- 分类器训练与融合 --------------------
def train_and_evaluate(features_train, features_test, labels_train, labels_test):
    """训练并评估分类器"""
    classifier = SVC(kernel='rbf', C=1, gamma=0.01, probability=True)
    classifier.fit(features_train, labels_train)
    predictions = classifier.predict(features_test)
    probabilities = classifier.predict_proba(features_test)
    report = classification_report(labels_test, predictions, output_dict=True)
    return report, probabilities

def late_fusion(probs_global, probs_local):
    """后期融合，通过加权平均概率计算最终预测"""
    combined_probs = (probs_global + probs_local) / 2
    final_predictions = np.argmax(combined_probs, axis=1)
    return final_predictions

# -------------------- 主程序 --------------------
# 加载数据
train_dirs = [
    "C:/Users/admin/OneDrive/Hong/subset_of_Ainimal_10/data/train/butterfly",
    "C:/Users/admin/OneDrive/Hong/subset_of_Ainimal_10/data/train/cat",
    "C:/Users/admin/OneDrive/Hong/subset_of_Ainimal_10/data/train/chicken",
    "C:/Users/admin/OneDrive/Hong/subset_of_Ainimal_10/data/train/cow",
    "C:/Users/admin/OneDrive/Hong/subset_of_Ainimal_10/data/train/dog",
    "C:/Users/admin/OneDrive/Hong/subset_of_Ainimal_10/data/train/sheep"
]

test_dirs = [
    "C:/Users/admin/OneDrive/Hong/subset_of_Ainimal_10/data/test/butterfly",
    "C:/Users/admin/OneDrive/Hong/subset_of_Ainimal_10/data/test/cat",
    "C:/Users/admin/OneDrive/Hong/subset_of_Ainimal_10/data/test/chicken",
    "C:/Users/admin/OneDrive/Hong/subset_of_Ainimal_10/data/test/cow",
    "C:/Users/admin/OneDrive/Hong/subset_of_Ainimal_10/data/test/dog",
    "C:/Users/admin/OneDrive/Hong/subset_of_Ainimal_10/data/test/sheep"
]

all_train_images, all_train_labels = load_all_data(train_dirs)
all_test_images, all_test_labels = load_all_data(test_dirs)

# 提取全局和局部特征
train_color_features = [extract_color_histogram(img) for img in all_train_images]
test_color_features = [extract_color_histogram(img) for img in all_test_images]

sift_descriptors = [extract_sift_features(img) for img in all_train_images]
bow_kmeans = KMeans(n_clusters=50, random_state=42).fit(np.vstack([desc for desc in sift_descriptors if desc is not None]))
train_bow_features = [create_bow_histogram(desc, bow_kmeans, 50) for desc in sift_descriptors]
test_bow_features = [create_bow_histogram(extract_sift_features(img), bow_kmeans, 50) for img in all_test_images]

# 早期融合特征
train_combined_features = [combine_features(c, b) for c, b in zip(train_color_features, train_bow_features)]
test_combined_features = [combine_features(c, b) for c, b in zip(test_color_features, test_bow_features)]

# 分类并评估
report_global, probs_global = train_and_evaluate(train_color_features, test_color_features, all_train_labels, all_test_labels)
print("Global Features Accuracy:", report_global["accuracy"])

report_local, probs_local = train_and_evaluate(train_bow_features, test_bow_features, all_train_labels, all_test_labels)
print("Local Features Accuracy:", report_local["accuracy"])

report_combined, _ = train_and_evaluate(train_combined_features, test_combined_features, all_train_labels, all_test_labels)
print("Early Fusion Accuracy:", report_combined["accuracy"])

# 后期融合
final_predictions = late_fusion(probs_global, probs_local)
report_fusion = classification_report(all_test_labels, final_predictions, output_dict=True)
print("Late Fusion Accuracy:", report_fusion["accuracy"])

# -------------------- 可视化 --------------------
def plot_results(reports, methods):
    metrics = ["accuracy"]
    fig, ax = plt.subplots(figsize=(8, 6))

    values = [
        reports[method]["accuracy"] * 100 for method in methods
    ]

    bars = ax.bar(methods, values, color=['blue', 'green', 'orange', 'red'])
    ax.set_title("Comparison of Accuracy", fontsize=16)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_ylim(min(values) - 5, 100)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 3, f'{value:.2f}%', ha='center', color='white', fontsize=12)

    plt.tight_layout()
    plt.show()

# 汇总结果
reports = {
    "Global Features": report_global,
    "Local Features": report_local,
    "Early Fusion": report_combined,
    "Late Fusion": report_fusion
}

plot_results(reports, ["Global Features", "Local Features", "Early Fusion", "Late Fusion"])
