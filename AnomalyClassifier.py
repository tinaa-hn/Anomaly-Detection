

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import os
import sys
import csv

csv.field_size_limit(sys.maxsize)

def load_time_series(file_path):
    data = pd.read_csv(file_path, header=None, sep='\t' if any(str(num) in file_path for num in [204, 205, 206, 207, 208, 225, 226, 242, 243]) else ',', engine='python')
    data = data.apply(pd.to_numeric, errors='coerce')
    return data.values.flatten()

def augment_data(X, y, multiplier=5):
    augmented_X = []
    augmented_y = []
    for i in range(len(X)):
        for _ in range(multiplier):
            augmented_X.append(X[i] + np.random.normal(0, 0.01, size=X[i].shape))  # Jittering
            augmented_y.append(y[i])
    return np.array(augmented_X), np.array(augmented_y)

def prepare_data(files, window_size=50, proximity=10):
    X = []
    y = []
    anomaly_count = Counter()

    for filename in files:
        ts = load_time_series(os.path.join(folder_path, filename))
        ts = (ts - np.min(ts)) / (np.max(ts) - np.min(ts))  
        
        parts = filename[:-4].split('_')
        anomaly_start = int(parts[-2])
        anomaly_end = int(parts[-1])

        if anomaly_start == anomaly_end:
            anomaly_end = anomaly_start + 1 

        anomaly_type = filename_to_anomaly_type.get(filename, "Unknown")
        start = max(0, anomaly_start - proximity)
        end = min(len(ts), anomaly_end + proximity)

        if end - start < window_size:
            start = max(0, end - window_size)  

        ts_segment = ts[start:end]

        features = [
            np.mean(ts_segment),
            np.std(ts_segment),
            np.max(ts_segment),
            np.min(ts_segment),
            np.percentile(ts_segment, 25),
            np.percentile(ts_segment, 75)
        ]

        if anomaly_type != "Unknown":
            X.append(features)
            y.append(anomaly_type)
            anomaly_count[anomaly_type] += 1

    return np.array(X), np.array(y), anomaly_count

folder_path = '/Users/tinahajinejad/Desktop/ALS new project/AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData'
anomaly_types_path = '/Users/tinahajinejad/Downloads/anomaly_types.csv'
anomaly_type_df = pd.read_csv(anomaly_types_path, delimiter=';')
filename_to_anomaly_type = dict(zip(anomaly_type_df['name'], anomaly_type_df['anomaly_type_2']))

all_files = os.listdir(folder_path)
_, _, anomaly_count = prepare_data(all_files)
valid_anomaly_types = {atype for atype, count in anomaly_count.items() if count >= 2}


models = []
weights = []

for anomaly_type in valid_anomaly_types:
    print(f"Training model for anomaly type: {anomaly_type}")
    
    type_files = [f for f in all_files if filename_to_anomaly_type.get(f) == anomaly_type]
    
    X, y, _ = prepare_data(type_files)
    
    #if the class is small, apply data augmentation
    if len(y) < 10:  
        X, y = augment_data(X, y, multiplier=(10 // len(y)))  # Increase multiplier based on the shortage

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    models.append((anomaly_type, model))
    weights.append(1 / len(y_train)) 

normalized_weights = [w / sum(weights) for w in weights]

#ensemble model with weighted voting
ensemble_clf = VotingClassifier(estimators=models, voting='soft', weights=normalized_weights)


X_test_all, y_test_all, _ = prepare_data(all_files)

ensemble_clf.fit(X_test_all, y_test_all)

ensemble_preds = ensemble_clf.predict(X_test_all)

labels = sorted(list(valid_anomaly_types))

print("Classification Report for Ensemble Model:")
print(classification_report(y_test_all, ensemble_preds, labels=labels, target_names=labels))


print("\nPredictions on Test Data:")
for i, (true_label, pred_label) in enumerate(zip(y_test_all, ensemble_preds)):
    print(f"File: {all_files[i]}, True Label: {true_label}, Predicted Label: {pred_label}")
