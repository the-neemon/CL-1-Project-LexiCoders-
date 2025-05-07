import pandas as pd
import numpy as np
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def train_crf_model(X_train, y_train):
    crf = CRF(algorithm='lbfgs',
              c1=0.1,
              c2=0.1,
              max_iterations=100,
              all_possible_transitions=True)
    crf.fit(X_train, y_train)
    return crf

def evaluate_model(crf, X_test, y_test):
    y_pred = crf.predict(X_test)
    report = flat_classification_report(y_test, y_pred)
    return report, y_pred

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def prepare_data(df):
    sentences = []
    current_sentence = []
    for _, row in df.iterrows():
        if pd.isna(row['Sent']):
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
        else:
            current_sentence.append((row['Word'], row['Tag'], row['Tag']))
    if current_sentence:
        sentences.append(current_sentence)
    return sentences

def main():
    training_files = [
        'training_data/train_datav1.csv',
        'training_data/shrish_data.csv',
        'training_data/naman_data_2.csv',
        'training_data/naman_data_3.csv',
        'training_data/train_datav2.csv',
        'training_data/annotatedData.csv',
        'training_data/naman_data_1.csv',
        'training_data/yash_data.csv'
    ]
    
    all_sentences = []
    for file in training_files:
        if os.path.exists(file):
            df = load_data(file)
            sentences = prepare_data(df)
            all_sentences.extend(sentences)
    
    X = [sent2features(s) for s in all_sentences]
    y = [sent2labels(s) for s in all_sentences]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    crf = train_crf_model(X_train, y_train)
    report, y_pred = evaluate_model(crf, X_test, y_test)
    
    all_labels = sorted(set([label for sent in y_test for label in sent]))
    plot_confusion_matrix([label for sent in y_test for label in sent],
                         [label for sent in y_pred for label in sent],
                         all_labels)
    
    print("\nClassification Report:")
    print(report)
    
    return crf, report

if __name__ == "__main__":
    main() 