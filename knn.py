import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score

def print_graph(y_test, pred=None, pred_proba=None):
    accuracy = accuracy_score(y_test , pred) # 정확도
    precision = precision_score(y_test , pred, average='macro') # 정밀도
    recall = recall_score(y_test , pred, average='macro') # 재현도
    f1 = f1_score(y_test,pred, average='macro')
    estimation=['accuracy','precision','recall']
    value=[accuracy, precision, recall]
    plt.bar(estimation, value)
    plt.title('k-NN')
    plt.show()
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
    F1: {3:.4f}'.format(accuracy, precision, recall, f1))

# Load Datasets
multi_data = pd.read_csv('open_multi.csv') #Open-World: Multi-Class Classification
binary_data = pd.read_csv('open_binary.csv') #Open-World: Binary Classification
closed_data = pd.read_csv('closed_dataset.csv') #Closed-World: Multi-Class Classification


#Open-World: Multi-Class Classification
X_multi = multi_data.drop('y', axis=1)
y_multi = multi_data['y']

#Open-World: Binary Classification
X_binary = binary_data.drop('y', axis=1)
y_binary = binary_data['y']

# Closed-World: Multi-Class Classification
X_closed = closed_data.drop('y', axis=1)
y_closed = closed_data['y']


# Manhattan distance 사용
pipeline_multi = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(weights='distance', p=1))  # p=1: Manhattan distance
])

pipeline_binary = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(weights='distance', p=1))
])

pipeline_closed = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(weights='distance', p=1))
])


param_grid = {
    'knn__n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]
}


# Open-World: Multi-Class Classification
X_multi_scaled = StandardScaler().fit_transform(X_multi)
X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(X_multi_scaled, y_multi, test_size=0.2,
                                                                            random_state=42)
# Open-World: Binary Classification
X_binary_scaled = StandardScaler().fit_transform(X_binary)
X_binary_train, X_binary_test, y_binary_train, y_binary_test = train_test_split(X_binary_scaled, y_binary,
                                                                                test_size=0.2, random_state=42)
# Closed-World: Multi-Class Classification
X_closed_scaled = StandardScaler().fit_transform(X_closed)
X_closed_train, X_closed_test, y_closed_train, y_closed_test = train_test_split(X_closed_scaled, y_closed,
                                                                                test_size=0.2, random_state=42)


# GridSearch
grid_search_multi = GridSearchCV(pipeline_multi, param_grid, cv=5, scoring='accuracy')
grid_search_multi.fit(X_multi_train, y_multi_train)

grid_search_closed = GridSearchCV(pipeline_closed, param_grid, cv=5, scoring='accuracy')
grid_search_closed.fit(X_closed_train, y_closed_train)

grid_search_binary = GridSearchCV(pipeline_binary, param_grid, cv=5, scoring='accuracy')
grid_search_binary.fit(X_binary_train, y_binary_train)


# Get the best k value 
best_k_multi = grid_search_multi.best_params_['knn__n_neighbors']
best_k_binary = grid_search_binary.best_params_['knn__n_neighbors']
best_k_closed = grid_search_closed.best_params_['knn__n_neighbors']

print("Best k for Open-World Multi-Class Classification:", best_k_multi)
print("Best k for Open-World Binary Classification:", best_k_binary)
print("Best k for Closed-World Classification:", best_k_closed)


# Evaluate model with the best k
y_multi_pred = grid_search_multi.predict(X_multi_test)
y_binary_pred = grid_search_binary.predict(X_binary_test)
y_closed_pred = grid_search_closed.predict(X_closed_test)

# Visualize evaluation metrics
print_graph(y_multi_test, y_multi_pred)
print_graph(y_closed_test, y_closed_pred)
print_graph(y_binary_test, y_binary_pred)
