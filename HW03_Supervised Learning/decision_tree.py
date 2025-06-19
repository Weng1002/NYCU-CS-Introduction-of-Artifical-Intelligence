import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from typing import List, Tuple

"""
Notice:
    1) You can't add any additional package
    2) You can add or remove any function "except" fit, _build_tree, predict
    3) You can ignore the suggested data type if you want
"""

class ConvNet(nn.Module): # Don't change this part!
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=300)

    def forward(self, x):
        x = self.model(x)
        return x
    
class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.data_size = X.shape[0]
        total_steps = 2 ** self.max_depth
        self.progress = tqdm(total=total_steps, desc="Growing tree", position=0, leave=True)
        self.tree = self._build_tree(X, y)
        self.progress.close()

    def _build_tree(self, X: pd.DataFrame, y: np.ndarray, depth: int):
        # (TODO) Grow the decision tree and return it
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()  
        
        best_feature, best_threshold = self._best_split(X, y)
        left_X, left_y, right_X, right_y = self._split_data(X, y, best_feature, best_threshold)
        
        left_node = self._build_tree(left_X, left_y, depth + 1)
        right_node = self._build_tree(right_X, right_y, depth + 1)
        
        return {"feature": best_feature, "threshold": best_threshold, "left": left_node, "right": right_node}

    def predict(self, X: pd.DataFrame)->np.ndarray:
        # (TODO) Call _predict_tree to traverse the decision tree to return the classes of the testing dataset
        return np.array([self._predict_tree(x, self.tree) for x in X.values])

    def _predict_tree(self, x, tree_node):
        # (TODO) Recursive function to traverse the decision tree
        if isinstance(tree_node, dict):
            feature = tree_node["feature"]
            threshold = tree_node["threshold"]
            if x[feature] <= threshold:
                return self._predict_tree(x, tree_node["left"])
            else:
                return self._predict_tree(x, tree_node["right"])
        return tree_node  

    def _split_data(self, X: pd.DataFrame, y: np.ndarray, feature_index: int, threshold: float):
        # (TODO) split one node into left and right node 
        left_mask = X.iloc[:, feature_index] <= threshold
        right_mask = ~left_mask
        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[right_mask], y[right_mask]
        return left_X, left_y, right_X, right_y
    

    def _best_split(self, X: pd.DataFrame, y: np.ndarray):
        # (TODO) Use Information Gain to find the best split for a dataset
        best_info_gain = -1
        best_feature_index = -1
        best_threshold = None
        
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X.iloc[:, feature_index])
            for threshold in thresholds:
                left_X, left_y, right_X, right_y = self._split_data(X, y, feature_index, threshold)
                
                # Calculate Information Gain
                info_gain = self._information_gain(y, left_y, right_y)
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature_index = feature_index
                    best_threshold = threshold
        
        return best_feature_index, best_threshold
    

    def _information_gain(self, parent_y, left_y, right_y):
        parent_entropy = self._entropy(parent_y)
        left_entropy = self._entropy(left_y)
        right_entropy = self._entropy(right_y)
        
        left_weight = len(left_y) / len(parent_y)
        right_weight = len(right_y) / len(parent_y)
        
        return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    def _entropy(self, y: np.ndarray)->float:
        # (TODO) Return the entropy
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

def get_features_and_labels(model: ConvNet, dataloader: DataLoader, device)->Tuple[pd.DataFrame, np.ndarray]:
    # (TODO) Use the model to extract features from the dataloader, return the features and labels
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Extracting features"):
            data, target = data.to(device), target.to(device)
            feature = model(data)  
            features.append(feature.cpu().numpy()) 
            labels.append(target.cpu().numpy())  

    # 將 features 由 list of arrays 變為一個單一的大 array
    features = np.concatenate(features, axis=0)  # shape: (total_samples, feature_dim)
    labels = np.concatenate(labels, axis=0)      # shape: (total_samples, )

    # 將 features 轉為 DataFrame
    features_df = pd.DataFrame(features)
    
    return features_df, labels


def get_features_and_paths(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    # (TODO) Use the model to extract features from the dataloader, return the features and path of the images
    model.eval()
    features = []
    paths = []
    
    with torch.no_grad():
        for data, target, path in tqdm(dataloader, desc="Extracting features with paths"):
            data = data.to(device)
            feature = model(data)  
            features.append(feature.cpu().numpy())  
            paths.extend(path)  
    
    return features, paths