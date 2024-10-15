# Author : Avin
# Python 3.8.0

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import sklearn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC


# Constants for dataset paths
DATASET_2010_2015 = 'data-2010-15'
DATASET_2020_2024 = 'data-2020-24'

class my_svm():
    def __init__(self):
        self.normalized_all_feature = None
        self.normalized_subset_feature = None
        self.data_order = None

    def preprocess(self, dataset_path):
        #load in order
        self.data_order = np.load(os.path.join(dataset_path, 'data_order.npy'))

        #load the negative data
        neg_features_historical = np.load(os.path.join(dataset_path, 'neg_features_historical.npy'))
        neg_features_main_timechange = np.load(os.path.join(dataset_path, 'neg_features_main_timechange.npy'))
        neg_features_maxmin = np.load(os.path.join(dataset_path, 'neg_features_maxmin.npy'))

        #load the positve data
        pos_features_historical = np.load(os.path.join(dataset_path, 'pos_features_historical.npy'))
        pos_features_main_timechange = np.load(os.path.join(dataset_path, 'pos_features_main_timechange.npy'))
        pos_features_maxmin = np.load(os.path.join(dataset_path, 'pos_features_maxmin.npy'))

        # set column names for neg_main_timechange_df (FSI and FSII) from 0 to 89
        neg_main_timechange_df = pd.DataFrame(neg_features_main_timechange).iloc[:, :90]
        neg_main_timechange_df.columns = range(90)  # Column names 0 to 89

        #set the column name for neg_historical_df (FSIII) to continue from 90
        neg_historical_df = pd.DataFrame(neg_features_historical, columns=[90])  # Column name 90

        # set colun names for neg_maxmin_df (FSIV) to continue from 91 onwards
        num_maxmin_columns = neg_features_maxmin.shape[1]
        neg_maxmin_df = pd.DataFrame(neg_features_maxmin)
        neg_maxmin_df.columns = range(91, 91 + num_maxmin_columns)

        # concatenate all negative features with properly aligned column names
        neg_features = pd.concat([neg_main_timechange_df, neg_historical_df, neg_maxmin_df], axis=1)
        neg_features['label'] = 0  # Add label column for negative data

        # Combine all positive features with increasing column names
        pos_main_timechange_df = pd.DataFrame(pos_features_main_timechange).iloc[:, :90]
        pos_main_timechange_df.columns = range(90)

        pos_historical_df = pd.DataFrame(pos_features_historical, columns=[90])

        num_pos_maxmin_columns = pos_features_maxmin.shape[1]
        pos_maxmin_df = pd.DataFrame(pos_features_maxmin)
        pos_maxmin_df.columns = range(91, 91 + num_pos_maxmin_columns)

        pos_features = pd.concat([pos_main_timechange_df, pos_historical_df, pos_maxmin_df], axis=1)
        pos_features['label'] = 1

        #concatenate neg and pos feature vertically
        combined_features = pd.concat([neg_features, pos_features], axis=0)

        #features are labels sperated
        features = combined_features.drop(columns=['label'])
        labels = combined_features['label']

        #convert all columns name to str
        features.columns = features.columns.astype(str)

        #normalize the features using StandardScaler
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)

        # Convert normalized features back to a DataFrame
        normalized_features_df = pd.DataFrame(normalized_features, columns=features.columns)

        self.normalized_all_feature = pd.concat([normalized_features_df, labels.reset_index(drop=True)], axis=1)



    def feature_creation(self, fs_value):
        # Define a mapping of feature set names to corresponding column slices
        feature_set_mapping = {
            'FSI': self.normalized_all_feature.columns[:18],    
            'FSII': self.normalized_all_feature.columns[18:90],
            'FSIII': self.normalized_all_feature.columns[90:91],
            'FSIV': self.normalized_all_feature.columns[91:109],
        }

        # eg 'FSI,FSII' becomes ['FSI', 'FSII']
        selected_feature_sets = fs_value.split(',')

        # Collect the corresponding columns based on the selected feature sets
        selected_columns = []
        for feature_set in selected_feature_sets:
            if feature_set.strip() in feature_set_mapping:
                selected_columns.extend(feature_set_mapping[feature_set.strip()])

        selected_columns.append('label')

        # only get slected coloums that it being specifies
        self.normalized_subset_feature = self.normalized_all_feature[selected_columns].to_numpy()


        # If the combination is 'FSI,FSIII', reorder according to data_order
        if fs_value == 'FSI,FSIII' and self.data_order is not None:
            # Reorder self.normalized_subset_feature according to self.data_order
            self.normalized_subset_feature = self.normalized_subset_feature[self.data_order]

    def cross_validation(self):
        k = 10 # number of folds
        skf = KFold(n_splits=k, shuffle=True, random_state=10)

        X = self.normalized_subset_feature[:, :-1]  # Features
        Y = self.normalized_subset_feature[:, -1]   # Labels

        tss_scores = []


        # initial confusion matrix components
        tn_total, fp_total, fn_total, tp_total = 0, 0, 0, 0

        # Perform cross-validation
        for train_index, test_index in skf.split(X, Y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            # train the model
            svm_classifier = self.training(X_train, y_train)

            y_pred = svm_classifier.predict(X_test)

            # Calculate confusion matrix for this fold
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            fn_total += fn
            tp_total += tp
            tn_total += tn
            fp_total += fp
            
            #calculate TSS for this fold
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            tss = sensitivity - false_alarm_rate
            tss_scores.append(tss)

        #calculat mean and standard deviation of TSS scores
        mean_tss = np.mean(tss_scores)
        std_tss = np.std(tss_scores)


        total_confusion_matrix = (tn_total, fp_total, fn_total, tp_total)
        return mean_tss, std_tss, tss_scores, total_confusion_matrix

    def training(self, X_train, y_train):
        #create an instance of the SVM classifier
        svm_classifier = SVC(kernel='rbf', C=1.2, gamma='scale')

        # Train the model using the features (X_train) and the labels (Y_train)
        svm_classifier.fit(X_train, y_train)
        return svm_classifier

    def tss(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()


        # Calculate TSS
        sensitivity = tp / (tp + fn) 
        false_alarm_rate = fp / (fp + tn)

        tss_score = sensitivity - false_alarm_rate
        return tss_score

# gets all combinations for FT combination
def generate_combinations(features):
    result = []

    def helper(current_combination, index):
        # Add current combination to the result if it's non-empty
        if current_combination:
            result.append(','.join(current_combination))

        # Iterate through remaining features to generate combinations
        for i in range(index, len(features)):
            helper(current_combination + [features[i]], i + 1)

    helper([], 0)

    return result


best_combination = None

def feature_experiment():
    global best_combination
    svm_instance = my_svm()

    # Preprocess the dataset to normalize the features
    svm_instance.preprocess(DATASET_2010_2015)

    # Define the feature set combinations to test
    feature_sets = [
        'FSI',
        'FSII',
        'FSIII',
        'FSIV'
    ]

    combinations = generate_combinations(feature_sets)

    best_tss = -float('inf') 

    #dictionaries to store TSS scores and confusion matrices
    tss_scores_dict = {}
    confusion_matrices_dict = {}


    # Loop through each feature combination
    for combination in combinations:
        svm_instance.feature_creation(combination)
        mean_tss, std_tss, tss_scores, total_confusion_matrix = svm_instance.cross_validation()
        print(f"({combination}) ~ Mean TSS : {mean_tss:.4f},   SD of TSS : {std_tss:.4f}")

        #stre TSS scores and confusion matrices
        tss_scores_dict[combination] = tss_scores
        confusion_matrices_dict[combination] = total_confusion_matrix

        # get max TSS
        if mean_tss >= best_tss:
            best_tss = mean_tss
            best_combination = combination

    plt.figure(figsize=(12, 6))
    num_lines = len(tss_scores_dict)
    colors = plt.cm.tab20(np.linspace(0, 1, num_lines))


    for idx, (combination, tss_scores) in enumerate(tss_scores_dict.items()):
        plt.plot(range(1, len(tss_scores) + 1), tss_scores, marker='o', label=combination, color=colors[idx])
    plt.title('TSS Scores for Various Feature Set Combinations')
    plt.xlabel('Fold')
    plt.ylabel('TSS Score')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.show()

    num_combinations = len(confusion_matrices_dict)
    cols = 4
    rows = (num_combinations + cols - 1) // cols 
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 2))
    axes = axes.flatten()

    for idx, (combination, cm_values) in enumerate(confusion_matrices_dict.items()):
        tn_total, fp_total, fn_total, tp_total = cm_values
        cm = np.array([[tn_total, fp_total],
                       [fn_total, tp_total]])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(ax=axes[idx], values_format='d', cmap='Blues', colorbar=False)
        axes[idx].set_title(f'CM: {combination}', fontsize=9.5)
        axes[idx].tick_params(axis='both', which='major', labelsize=8)
        axes[idx].set_xlabel('Predicted', fontsize=8)
        axes[idx].set_ylabel('Actual', fontsize=8)

    # Hide any unused subplots
    for ax in axes[num_combinations:]:
        ax.axis('off')


    plt.tight_layout()
    plt.show()

    print()
    print(f"Best feature set combination is: {best_combination} with Mean TSS: {best_tss:.4f}\n")

def data_experiment():
    global best_combination
    svm_instance = my_svm()

    # Dictionaries to store TSS scores and confusion matrices
    datasets = {'2010-2015': DATASET_2010_2015,
                '2020-2024': DATASET_2020_2024}
    tss_scores_dict = {}
    confusion_matrices_dict = {}
    mean_tss_dict = {}

    for dataset_name, dataset_path in datasets.items():
        print(f"Running experiment on {dataset_name} dataset...")
        svm_instance.preprocess(dataset_path)
        svm_instance.feature_creation(best_combination)
        mean_tss, std_tss, tss_scores, total_confusion_matrix = svm_instance.cross_validation()
        mean_tss_dict[dataset_name] = mean_tss
        print(f"({dataset_name}) ~ Mean TSS : {mean_tss:.4f}, SD of TSS : {std_tss:.4f}")
        tss_scores_dict[dataset_name] = tss_scores
        confusion_matrices_dict[dataset_name] = total_confusion_matrix


    plt.figure(figsize=(10, 7))
    colors = ['blue', 'red']
    for idx, (dataset_name, tss_scores) in enumerate(tss_scores_dict.items()):
        plt.plot(range(1, len(tss_scores) + 1), tss_scores, marker='o', label=dataset_name, color=colors[idx])
    plt.title('TSS Scores for Different Datasets with Best FT Combination')
    plt.xlabel('Fold')
    plt.ylabel('TSS Score')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.84, 1])
    plt.show()

    print()
    if mean_tss_dict['2020-2024'] > mean_tss_dict['2010-2015']:
        print("The 2020-2024 dataset has achieved better performance.")
    else:
        print("The 2010-2015 dataset has achieved better performance.")

feature_experiment()
data_experiment()
