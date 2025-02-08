import glob

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def meta_load():
    def load_data(file_pattern):
        data = {}
        files = glob.glob(file_pattern)
        for file in files:
            batch_data = np.load(file, allow_pickle=True)
            for key in batch_data:
                if key not in data:
                    data[key] = []
                data[key].append(batch_data[key])
        for key in data:
            data[key] = np.concatenate(data[key], axis=0)
        print('load all, no memory problem')
        return data
    return load_data('output_npz/batch_*.npz')


data = meta_load()

matrix_shape = data['matrix'].shape
num_samples = matrix_shape[0]
num_features = np.prod(matrix_shape[1:])

X = data['matrix'].reshape(num_samples, num_features)

label_keys = [key for key in data.keys() if key.startswith('label_')]

classifiers = {}
metrics = {}

for label_key in label_keys:
    y = np.argmax(data[label_key], axis=1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate on the test set
    y_test_pred = rf.predict(X_test)
    y_train_pred = rf.predict(X_train)

    # Calculate metrics for the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

    # Calculate metrics for the training set
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)

    # Store the metrics
    classifiers[label_key] = rf
    metrics[label_key] = {
        'train': {
            'accuracy': train_accuracy,
            'precision': train_precision,
            'recall': train_recall,
            'f1': train_f1
        },
        'test': {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1
        }
    }

    # Print metrics for the current label
    print(f"\nMetrics for {label_key}:")
    print(f"  Train - Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
    print(f"  Test  - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    model_filename = f"decisionModel/{label_key}_random_forest_model.pkl"
    joblib.dump(rf, model_filename)
    print(f"Model for {label_key} saved as {model_filename}")

# Print summary of all metrics
print("\nSummary of Metrics:")
for label_key, metric in metrics.items():
    print(f"\n{label_key}:")
    print(f"  Train - Accuracy: {metric['train']['accuracy']:.4f}, Precision: {metric['train']['precision']:.4f}, Recall: {metric['train']['recall']:.4f}, F1: {metric['train']['f1']:.4f}")
    print(f"  Test  - Accuracy: {metric['test']['accuracy']:.4f}, Precision: {metric['test']['precision']:.4f}, Recall: {metric['test']['recall']:.4f}, F1: {metric['test']['f1']:.4f}")
