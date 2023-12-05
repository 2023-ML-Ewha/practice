import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Load datasets
monitored_data = pd.read_csv('dataset.csv')
unmonitored_data = pd.read_csv('unmon_dataset.csv')

# Closed-World Experiments

## Multi-Class Classification
# Assuming each monitored website has 10 subpages observed 20 times each
# This results in 95 * 10 * 20 = 19,000 rows
num_monitored_websites = 95
num_subpages_per_website = 10
num_observations_per_subpage = 20

# Select features and labels for closed-world multi-class classification
X_closed = monitored_data.drop(['y'], axis=1)  # Assuming 'y' is the column with labels
y_closed = monitored_data['y']

# Impute missing values in the features
imputer = SimpleImputer(strategy='mean')
X_closed_imputed = pd.DataFrame(imputer.fit_transform(X_closed), columns=X_closed.columns)

# Split the data into training and testing sets
X_train_closed, X_test_closed, y_train_closed, y_test_closed = train_test_split(X_closed_imputed, y_closed, test_size=0.2, random_state=42)

# Initialize and train the model for closed-world multi-class classification
closed_classifier = KNeighborsClassifier(n_neighbors=5)
closed_classifier.fit(X_train_closed, y_train_closed)

# Make predictions on the test set for closed-world multi-class classification
y_pred_closed = closed_classifier.predict(X_test_closed)

# Evaluate the performance of the closed-world multi-class classification model
accuracy_closed = accuracy_score(y_test_closed, y_pred_closed)
print(f'Closed-World Multi-Class Classification Accuracy: {accuracy_closed * 100:.2f}%')

# Classification Report for Closed-World Multi-Class
print('Classification Report for Closed-World Multi-Class:')
print(classification_report(y_test_closed, y_pred_closed))

# Open-World Experiments

## Binary Classification
# Assign labels for binary classification
monitored_data['label_binary'] = 1
unmonitored_data['label_binary'] = -1

# Concatenate the datasets for binary classification
binary_data = pd.concat([monitored_data, unmonitored_data], ignore_index=True)

# Select features and labels for binary classification
X_binary = binary_data.drop(['label_binary'], axis=1)
y_binary = binary_data['label_binary']

# Impute missing values in the features
X_binary_imputed = pd.DataFrame(imputer.fit_transform(X_binary), columns=X_binary.columns)

# Split the data into training and testing sets for binary classification
X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(X_binary_imputed, y_binary, test_size=0.2, random_state=42)

# Initialize and train the model for binary classification
binary_classifier = KNeighborsClassifier(n_neighbors=5)
binary_classifier.fit(X_train_binary, y_train_binary)

# Make predictions on the test set for binary classification
y_pred_binary = binary_classifier.predict(X_test_binary)

# Evaluate the performance of the binary classification model
accuracy_binary = accuracy_score(y_test_binary, y_pred_binary)
print(f'Binary Classification Accuracy: {accuracy_binary * 100:.2f}%')

# Multi-Class Classification
# Assign labels for multi-class classification
labels_multi = [label for label in range(num_monitored_websites) for _ in range(num_subpages_per_website * num_observations_per_subpage)]
monitored_data['label_multi'] = labels_multi
unmonitored_data['label_multi'] = -1

# Concatenate the datasets for multi-class classification
multi_data = pd.concat([monitored_data, unmonitored_data], ignore_index=True)

# Select features and labels for multi-class classification
X_multi = multi_data.drop(['label_multi'], axis=1)
y_multi = multi_data['label_multi']

# Impute missing values in the features
X_multi_imputed = pd.DataFrame(imputer.fit_transform(X_multi), columns=X_multi.columns)

# Split the data into training and testing sets for multi-class classification
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi_imputed, y_multi, test_size=0.2, random_state=42)

# Initialize and train the model for multi-class classification
multi_classifier = KNeighborsClassifier(n_neighbors=5)
multi_classifier.fit(X_train_multi, y_train_multi)

# Make predictions on the test set for multi-class classification
y_pred_multi = multi_classifier.predict(X_test_multi)

# Evaluate the performance of the multi-class classification model
accuracy_multi = accuracy_score(y_test_multi, y_pred_multi)
print(f'Multi-Class Classification Accuracy: {accuracy_multi * 100:.2f}%')

# Classification Report for Multi-Class
print('Classification Report for Multi-Class:')
print(classification_report(y_test_multi, y_pred_multi, target_names=[str(label) for label in range(num_monitored_websites)] + ['-1']))
