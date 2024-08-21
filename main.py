#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, ParameterGrid
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict
from sklearn.base import clone


# In[ ]:


tf.random.set_seed(3)
np.random.seed(3)
random.seed(3)


# In[ ]:


def create_model(input_dim, neurons=32, dropout=0.2):
    inputs = Input(shape=(input_dim,))
    x = Dense(neurons, activation='relu')(inputs)
    x = Dropout(dropout)(x)
    x = Dense(neurons // 2, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[ ]:


def cross_val_grid_search(X, y, param_grid, input_dim, n_splits=5, validation_size=0.2):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=3)
    best_scores = []
    best_params_list = []
    test_metrics = []
    for train_val_index, test_index in cv.split(X):
        X_train_val, X_test = X[train_val_index], X[test_index]
        y_train_val, y_test = np.array(y)[train_val_index], np.array(y)[test_index]
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=validation_size, random_state=3)
        best_score = 0
        best_params = None
        best_model = None
        for params in ParameterGrid(param_grid):
            neurons = params['neurons']
            dropout = params['dropout']
            batch_size = params['batch_size']
            epochs = params['epochs']
            smote = SMOTE(random_state=3)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            model = create_model(input_dim=input_dim, neurons=neurons, dropout=dropout)
            model.fit(X_train_smote, y_train_smote, validation_data=(X_val, y_val),
                      batch_size=batch_size, epochs=epochs, verbose=0)
            val_predictions = model.predict(X_val)
            val_predictions = (val_predictions > 0.5).astype(int)
            val_f1_score_class_1 = f1_score(y_val, val_predictions, pos_label=1)
            if val_f1_score_class_1 > best_score:
                best_score = val_f1_score_class_1
                best_params = params
                best_model = model
        best_scores.append(best_score)
        best_params_list.append(best_params)
        test_predictions = best_model.predict(X_test)
        test_predictions = (test_predictions > 0.5).astype(int)
        test_acc = accuracy_score(y_test, test_predictions)
        test_prec = precision_score(y_test, test_predictions)
        test_rec = recall_score(y_test, test_predictions)
        test_f1 = f1_score(y_test, test_predictions)
        test_metrics.append({
            'accuracy': test_acc,
            'precision': test_prec,
            'recall': test_rec,
            'f1_score': test_f1
        })

    return best_params_list, best_scores, test_metrics


# In[ ]:


def evaluate_params(X, y, params_list, input_dim, n_splits=5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=3)
    param_f1_scores = defaultdict(list)
    param_metrics = defaultdict(list)
    for params in params_list:
        f1_scores = []
        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
        for train_index, val_index in cv.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = np.array(y)[train_index], np.array(y)[val_index]
            smote = SMOTE(random_state=3)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            model = create_model(input_dim=input_dim, neurons=params['neurons'], dropout=params['dropout'])
            model.fit(X_train_smote, y_train_smote, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
            val_predictions = model.predict(X_val)
            val_predictions = (val_predictions > 0.5).astype(int)
            metrics['accuracy'].append(accuracy_score(y_val, val_predictions))
            metrics['precision'].append(precision_score(y_val, val_predictions))
            metrics['recall'].append(recall_score(y_val, val_predictions))
            f1_scores.append(f1_score(y_val, val_predictions))
        mean_accuracy = np.mean(metrics['accuracy'])
        mean_precision = np.mean(metrics['precision'])
        mean_recall = np.mean(metrics['recall'])
        mean_f1_score = np.mean(f1_scores)
        param_f1_scores[tuple(params.items())].append(mean_f1_score)
        param_metrics[tuple(params.items())].append({
            'accuracy': mean_accuracy,
            'precision': mean_precision,
            'recall': mean_recall,
            'f1_score': mean_f1_score
        })
    mean_param_f1_scores = {k: np.mean(v) for k, v in param_f1_scores.items()}
    best_params_tuple = max(mean_param_f1_scores, key=mean_param_f1_scores.get)
    best_params = dict(best_params_tuple)
    best_metrics = param_metrics[best_params_tuple]

    return best_params, best_metrics


# In[ ]:


def cross_val_grid_search_classifiers(classifiers, param_grids, X, y, n_splits=5, validation_size=0.2):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=3)
    best_params_list = []
    best_scores = []
    test_metrics = []
    for clf_name, clf in classifiers.items():
        param_grid = param_grids.get(clf_name, {})
        for train_val_index, test_index in cv.split(X):
            X_train_val, X_test = X[train_val_index], X[test_index]
            y_train_val, y_test = np.array(y)[train_val_index], np.array(y)[test_index]
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=validation_size, random_state=3)
            smote = SMOTE(random_state=3)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            best_score = 0
            best_params = None
            best_clf = None
            for params in ParameterGrid(param_grid):
                current_clf = clf
                current_clf.set_params(**params)
                current_clf.fit(X_train_smote, y_train_smote)
                val_predictions = current_clf.predict(X_val)
                val_f1_score = f1_score(y_val, val_predictions, pos_label=1)
                if val_f1_score > best_score:
                    best_score = val_f1_score
                    best_params = params
                    best_clf = current_clf
            best_params_list.append({clf_name: best_params})
            best_scores.append(best_score)
            test_predictions = best_clf.predict(X_test)
            test_acc = accuracy_score(y_test, test_predictions)
            test_prec = precision_score(y_test, test_predictions)
            test_rec = recall_score(y_test, test_predictions)
            test_f1 = f1_score(y_test, test_predictions)
            test_metrics.append({
                'Classifier': clf_name,
                'Accuracy': test_acc,
                'Precision': test_prec,
                'Recall': test_rec,
                'F1 Score': test_f1
            })

    return best_params_list, best_scores, test_metrics


# In[ ]:


def evaluate_params_classifiers(X, y, params_list, classifiers, n_splits=5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=3)
    best_params = {}
    best_metrics = {}
    for clf_name in classifiers.keys():
        clf_best_f1_score = -1
        clf_best_params = None
        clf_best_metrics = None
        for param_dict in params_list:
            if clf_name in param_dict:
                clf_params = param_dict[clf_name]
                f1_scores = []
                metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
                for train_index, val_index in cv.split(X):
                    X_train, X_val = X[train_index], X[val_index]
                    y_train, y_val = np.array(y)[train_index], np.array(y)[val_index]
                    smote = SMOTE(random_state=3)
                    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
                    clf = clone(classifiers[clf_name])
                    clf.set_params(**clf_params)
                    clf.fit(X_train_smote, y_train_smote)
                    val_predictions = clf.predict(X_val)
                    metrics['accuracy'].append(accuracy_score(y_val, val_predictions))
                    metrics['precision'].append(precision_score(y_val, val_predictions))
                    metrics['recall'].append(recall_score(y_val, val_predictions))
                    f1_scores.append(f1_score(y_val, val_predictions, pos_label=1))
                mean_accuracy = np.mean(metrics['accuracy'])
                mean_precision = np.mean(metrics['precision'])
                mean_recall = np.mean(metrics['recall'])
                mean_f1_score = np.mean(f1_scores)
                if mean_f1_score > clf_best_f1_score:
                    clf_best_f1_score = mean_f1_score
                    clf_best_params = clf_params
                    clf_best_metrics = {
                        'accuracy': mean_accuracy,
                        'precision': mean_precision,
                        'recall': mean_recall,
                        'f1_score': mean_f1_score
                    }
        if clf_best_params is not None:
            best_params[clf_name] = clf_best_params
            best_metrics[clf_name] = clf_best_metrics
    return best_params, best_metrics


# In[ ]:


def get_classifiers(best_params):
    classifiers = {
        'RandomForest': RandomForestClassifier(**best_params.get('RandomForest', {})),
        'LogisticRegression': LogisticRegression(**best_params.get('LogisticRegression', {})),
        'SVC': SVC(probability=True, **best_params.get('SVC', {})),
        'ExtraTrees': ExtraTreesClassifier(**best_params.get('ExtraTrees', {})),
        'DecisionTree': DecisionTreeClassifier(**best_params.get('DecisionTree', {})),
        'NaiveBayes': GaussianNB(**best_params.get('NaiveBayes', {})), 
        'GradientBoosting': GradientBoostingClassifier(**best_params.get('GradientBoosting', {})),
        'KNN': KNeighborsClassifier(**best_params.get('KNN', {}))
    }
    return classifiers


# In[ ]:


class MLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, neurons=32, dropout=0.2, batch_size=32, epochs=100):
        self.input_dim = input_dim
        self.neurons = neurons
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
    def fit(self, X, y):
        self.model = create_model(input_dim=self.input_dim, neurons=self.neurons, dropout=self.dropout)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self
    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)
    def predict_proba(self, X):
        proba = self.model.predict(X)
        return np.hstack((1 - proba, proba))


# In[ ]:


def soft_voting_combinations(X, y, classifiers, n_splits=5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=3)
    results = defaultdict(list)
    classifier_names = list(classifiers.keys())
    all_combinations = []
    for r in range(3, len(classifier_names) + 1):
        all_combinations.extend(combinations(classifier_names, r))
    for comb in all_combinations:
        clf_list = [(name, classifiers[name]) for name in comb]
        ensemble_clf = VotingClassifier(estimators=clf_list, voting='soft')
        f1_scores = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        for train_index, test_index in cv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
            smote = SMOTE(random_state=3)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            ensemble_clf.fit(X_train_smote, y_train_smote)
            y_pred = ensemble_clf.predict(X_test)
            f1_scores.append(f1_score(y_test, y_pred, pos_label=1))
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred))
            recall_scores.append(recall_score(y_test, y_pred))
        mean_f1_score = np.mean(f1_scores)
        mean_accuracy = np.mean(accuracy_scores)
        mean_precision = np.mean(precision_scores)
        mean_recall = np.mean(recall_scores)
        results['combination'].append(comb)
        results['mean_f1_score'].append(mean_f1_score)
        results['mean_accuracy'].append(mean_accuracy)
        results['mean_precision'].append(mean_precision)
        results['mean_recall'].append(mean_recall)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='mean_f1_score', ascending=False)
    return results_df


# In[ ]:


def main():
    dataset_wbcd = pd.read_csv("breast_cancer.csv")
    dataset_wbcd = dataset_wbcd.drop(['Unnamed: 32', 'id'], axis=1)
    dataset_wbcd['diagnosis'] = dataset_wbcd['diagnosis'].replace({'B': 0, 'M': 1})
    y_wbcd = dataset_wbcd['diagnosis']
    X_wbcd = dataset_wbcd.drop('diagnosis', axis=1)
    
    dataset_heart = pd.read_csv("heart.csv")
    y_heart = dataset_heart['HeartDisease']
    X_heart = pd.get_dummies(dataset_heart.drop('HeartDisease', axis=1), drop_first=True)
    
    dataset_stroke = pd.read_csv("stroke.csv")
    dataset_stroke = dataset_stroke.dropna().drop('id', axis=1)
    y_stroke = dataset_stroke['stroke']
    X_stroke = pd.get_dummies(dataset_stroke.drop('stroke', axis=1), drop_first=True)
    
    scaler = StandardScaler()
    X_wbcd = scaler.fit_transform(X_wbcd)
    X_heart = scaler.fit_transform(X_heart)
    X_stroke = scaler.fit_transform(X_stroke)
    classifiers = {
    'RandomForest': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
    'SVC': SVC(),
    'ExtraTrees': ExtraTreesClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'NaiveBayes': GaussianNB(),
    'GradientBoosting': GradientBoostingClassifier(),
    'KNN': KNeighborsClassifier()}

    param_grid_mlp = {
    'neurons': [32, 64, 128],
    'dropout': [0.2, 0.4],
    'batch_size': [32, 64],
    'epochs': [100, 200]}
    
    param_grids = {
    'RandomForest': {'n_estimators': [100, 500, 1000]},
    'LogisticRegression': {'max_iter': [1000, 2000, 3000]},
    'SVC': {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10]},
    'ExtraTrees': {'n_estimators': [100, 500, 1000]},
    'DecisionTree': {'max_depth': [None, 10, 50, 100]},
    'NaiveBayes': {},
    'GradientBoosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5], 'max_depth': [3, 5, 7]},
    'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    }
    
    best_params_list_wbcd, best_scores_wbcd, test_metrics_wbcd = cross_val_grid_search(X_wbcd, y_wbcd, param_grid_mlp, X_wbcd.shape[1])
    input_dim_wbcd = X_wbcd.shape[1]
    unique_params_wbcd = [dict(t) for t in {tuple(d.items()) for d in best_params_list_wbcd}]
    best_params_wbcd, best_metrics = evaluate_params(X_wbcd, y_wbcd, unique_params_wbcd, input_dim_wbcd)

    best_params_list_heart, best_scores_heart, test_metrics_heart = cross_val_grid_search(X_heart, y_heart, param_grid_mlp, X_heart.shape[1])
    input_dim_heart = X_heart.shape[1]
    unique_params_heart = [dict(t) for t in {tuple(d.items()) for d in best_params_list_heart}]
    best_params_heart, best_metrics_heart = evaluate_params(X_heart, y_heart, unique_params_heart, input_dim_heart)
    
    best_params_list_stroke, best_scores_stroke, test_metrics_stroke = cross_val_grid_search(X_stroke, y_stroke, param_grid_mlp, X_stroke.shape[1])
    input_dim_stroke = X_stroke.shape[1]
    unique_params_stroke = [dict(t) for t in {tuple(d.items()) for d in best_params_list_stroke}]
    best_params_stroke, best_metrics_stroke = evaluate_params(X_stroke, y_stroke, unique_params_stroke, input_dim_stroke)
    
    best_params_list_classifiers_wbcd, best_scores_classifiers_wbcd, test_metrics_classifiers_wbcd = cross_val_grid_search_classifiers(classifiers, param_grids, X_wbcd, y_wbcd)
    best_params_wbcd_classifiers, best_metrics_wbcd_classifiers = evaluate_params_classifiers(X_wbcd, y_wbcd, best_params_list_classifiers_wbcd, classifiers)
    
    print("\nBCWD - parameters found by grid search for each classifier:")
    print("MLP:", best_params_wbcd)
    print("MLP metrics:")
    print(best_metrics) 
    print("Random Forest:", best_params_wbcd_classifiers['RandomForest'])
    print("Random Forest metrics:")
    print(best_metrics_wbcd_classifiers['RandomForest'])
    print("Gradient Boosting:", best_params_wbcd_classifiers['GradientBoosting'])
    print("Gradient Boosting metrics:")
    print(best_metrics_wbcd_classifiers['GradientBoosting'])
    print("Logistic Regression:", best_params_wbcd_classifiers['LogisticRegression'])
    print("Logistic Regression metrics:")
    print(best_metrics_wbcd_classifiers['LogisticRegression'])
    print("SVM:", best_params_wbcd_classifiers['SVC'])
    print("SVM metrics:")
    print(best_metrics_wbcd_classifiers['SVC'])
    print("KNN:", best_params_wbcd_classifiers['KNN'])
    print("KNN metrics:")
    print(best_metrics_wbcd_classifiers['KNN'])
    print("Extra Trees:", best_params_wbcd_classifiers['ExtraTrees'])
    print("Extra Trees metrics:")
    print(best_metrics_wbcd_classifiers['ExtraTrees'])
    print("Decision Tree:", best_params_wbcd_classifiers['DecisionTree'])
    print("Decision Tree metrics:")
    print(best_metrics_wbcd_classifiers['DecisionTree'])

    
    best_params_list_classifiers_heart, best_scores_classifiers_heart, test_metrics_classifiers_heart = cross_val_grid_search_classifiers(classifiers, param_grids, X_heart, y_heart)
    best_params_heart_classifiers, best_metrics_heart_classifiers = evaluate_params_classifiers(X_heart, y_heart, best_params_list_classifiers_heart, classifiers)
    
    print("\nHFPD - parameters found by grid search for each classifier:")
    print("MLP:", best_params_heart)
    print("MLP metrics:")
    print(best_metrics_heart)
    print("Random Forest:", best_params_heart_classifiers['RandomForest'])
    print("Random Forest metrics:")
    print(best_metrics_heart_classifiers['RandomForest'])
    print("Gradient Boosting:", best_params_heart_classifiers['GradientBoosting'])
    print("Gradient Boosting metrics:")
    print(best_metrics_heart_classifiers['GradientBoosting'])
    print("Logistic Regression:", best_params_heart_classifiers['LogisticRegression'])
    print("Logistic Regression metrics:")
    print(best_metrics_heart_classifiers['LogisticRegression'])
    print("SVM:", best_params_heart_classifiers['SVC'])
    print("SVM metrics:")
    print(best_metrics_heart_classifiers['SVC'])
    print("KNN:", best_params_heart_classifiers['KNN'])
    print("KNN metrics:")
    print(best_metrics_heart_classifiers['KNN'])
    print("Extra Trees:", best_params_heart_classifiers['ExtraTrees'])
    print("Extra Trees metrics:")
    print(best_metrics_heart_classifiers['ExtraTrees'])
    print("Decision Tree:", best_params_heart_classifiers['DecisionTree'])
    print("Decision Tree metrics:")
    print(best_metrics_heart_classifiers['DecisionTree'])
    
    best_params_list_classifiers_stroke, best_scores_classifiers_stroke, test_metrics_classifiers_stroke = cross_val_grid_search_classifiers(classifiers, param_grids, X_stroke, y_stroke)
    best_params_stroke_classifiers, best_metrics_stroke_classifiers = evaluate_params_classifiers(X_stroke, y_stroke, best_params_list_classifiers_stroke, classifiers)
    
    print("\nSPD - parameters found by grid search for each classifier:")
    print("MLP:", best_params_stroke)
    print("MLP metrics:")
    print(best_metrics_stroke)
    print("Random Forest:", best_params_stroke_classifiers['RandomForest'])
    print("Random Forest metrics:")
    print(best_metrics_stroke_classifiers['RandomForest'])
    print("Gradient Boosting:", best_params_stroke_classifiers['GradientBoosting'])
    print("Gradient Boosting metrics:")
    print(best_metrics_stroke_classifiers['GradientBoosting'])
    print("Logistic Regression:", best_params_stroke_classifiers['LogisticRegression'])
    print("Logistic Regression metrics:")
    print(best_metrics_stroke_classifiers['LogisticRegression'])
    print("SVM:", best_params_stroke_classifiers['SVC'])
    print("SVM metrics:")
    print(best_metrics_stroke_classifiers['SVC'])
    print("KNN:", best_params_stroke_classifiers['KNN'])
    print("KNN metrics:")
    print(best_metrics_stroke_classifiers['KNN'])
    print("Extra Trees:", best_params_stroke_classifiers['ExtraTrees'])
    print("Extra Trees metrics:")
    print(best_metrics_stroke_classifiers['ExtraTrees'])
    print("Decision Tree:", best_params_stroke_classifiers['DecisionTree'])
    print("Decision Tree metrics:")
    print(best_metrics_stroke_classifiers['DecisionTree'])
    
    classifiers_wbcd = get_classifiers(best_params_wbcd_classifiers)
    mlp_wrapper = MLPWrapper(input_dim=input_dim_wbcd, **best_params_wbcd)
    classifiers_wbcd['MLP'] = mlp_wrapper
    
    results_wbcd = soft_voting_combinations(X_wbcd, y_wbcd, classifiers_wbcd)
    pd.set_option('display.max_colwidth', None)
    print("Results of best ensemble search for BCWD: ")
    print(results_wbcd.head(5))
    
    classifiers_stroke = get_classifiers(best_params_stroke_classifiers)
    mlp_wrapper_stroke = MLPWrapper(input_dim=input_dim_stroke, **best_params_stroke)
    classifiers_stroke['MLP'] = mlp_wrapper_stroke
    
    results_stroke = soft_voting_combinations(X_stroke, y_stroke, classifiers_stroke)
    print("Results of best ensemble search for SPD: ")
    print(results_stroke.head(5))
    
    classifiers_heart = get_classifiers(best_params_heart_classifiers)
    mlp_wrapper_heart = MLPWrapper(input_dim=input_dim_heart, **best_params_heart)
    classifiers_heart['MLP'] = mlp_wrapper_heart
    
    results_heart = soft_voting_combinations(X_heart, y_heart, classifiers_heart)
    print(results_heart.head(5))


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:




