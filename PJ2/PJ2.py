import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix

# -------------------- 数据预处理与增强 --------------------
def preprocess_image(image):
    """基本图像预处理：调整大小、去噪、归一化"""
    image_resized = cv2.resize(image, (224, 224))  # 调整图像大小
    image_blurred = cv2.GaussianBlur(image_resized, (5, 5), 0)  # 高斯模糊去噪
    image_normalized = image_blurred / 255.0  # 归一化到[0, 1]
    
    # 确保数据类型为 uint8
    image_uint8 = (image_normalized * 255).astype(np.uint8)
    
    return image_uint8

def augment_image(image):
    """简单的数据增强：随机旋转和翻转"""
    if random.random() > 0.5:
        image = cv2.flip(image, 1)  # 水平翻转
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return image

def load_images_from_folder(folder, label, augment=False):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = preprocess_image(img)  # 数据预处理
            if augment:
                img = augment_image(img)  # 数据增强
            images.append(img)
            labels.append(label)
    return images, labels

def load_all_data(data_dirs, augment=False):
    all_images = []
    all_labels = []
    for label, folder in enumerate(data_dirs):
        images, labels = load_images_from_folder(folder, label, augment)
        all_images.extend(images)
        all_labels.extend(labels)
    return all_images, all_labels

# -------------------- 特征提取 --------------------
def extract_color_histogram(image):
    """提取颜色直方图（全局特征）"""
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image_hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_sift_features(image):
    """提取 SIFT 局部特征"""
    sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.04, edgeThreshold=10)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def create_bow_histogram(descriptors, bow_kmeans, voc_size):
    """使用 BoW 模型将 SIFT 特征转换为特征向量"""
    if descriptors is None:
        return np.zeros(voc_size)
    words = bow_kmeans.predict(descriptors)
    histogram = np.zeros(voc_size)
    for word in words:
        histogram[word] += 1
    return histogram

# -------------------- 网格搜索与分类器优化 --------------------
param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}

param_grid_tree = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

def train_and_evaluate_with_grid_search(features_train, features_test, labels_train, labels_test):
    classifier_results = {}
    
    # SVM
    grid_search_svm = GridSearchCV(SVC(probability=True), param_grid_svm, cv=5, n_jobs=-1)
    grid_search_svm.fit(features_train, labels_train)
    best_svm = grid_search_svm.best_estimator_
    predictions_svm = best_svm.predict(features_test)
    report_svm = classification_report(labels_test, predictions_svm, output_dict=True)
    classifier_results["SVM"] = report_svm
    
    # KNN
    grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, n_jobs=-1)
    grid_search_knn.fit(features_train, labels_train)
    best_knn = grid_search_knn.best_estimator_
    predictions_knn = best_knn.predict(features_test)
    report_knn = classification_report(labels_test, predictions_knn, output_dict=True)
    classifier_results["KNN"] = report_knn
    
    # Decision Tree
    grid_search_tree = GridSearchCV(DecisionTreeClassifier(), param_grid_tree, cv=5, n_jobs=-1)
    grid_search_tree.fit(features_train, labels_train)
    best_tree = grid_search_tree.best_estimator_
    predictions_tree = best_tree.predict(features_test)
    report_tree = classification_report(labels_test, predictions_tree, output_dict=True)
    classifier_results["Decision Tree"] = report_tree
    
    return classifier_results

# -------------------- 主程序 --------------------
# 加载训练和测试数据
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

# 加载数据集
all_train_images, all_train_labels = load_all_data(train_dirs, augment=True)
all_test_images, all_test_labels = load_all_data(test_dirs, augment=False)

# 提取特征
train_color_features = [extract_color_histogram(img) for img in all_train_images]
test_color_features = [extract_color_histogram(img) for img in all_test_images]

# 执行网格搜索并评估分类器
results = train_and_evaluate_with_grid_search(train_color_features, test_color_features, all_train_labels, all_test_labels)

# 打印分类器结果
def print_results(results):
    for clf_name, result in results.items():
        print(f"\n{clf_name} Results:")
        print(f"Accuracy: {result['accuracy']}")
        print(f"Precision: {result['weighted avg']['precision']}")
        print(f"Recall: {result['weighted avg']['recall']}")
        print(f"F1-Score: {result['weighted avg']['f1-score']}")

print_results(results)
