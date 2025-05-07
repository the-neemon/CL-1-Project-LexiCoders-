from model import (
    load_all_training_data,
    load_data_from_csv,
    sent2features,
    sent2labels,
    train_model,
    evaluate_model
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

def plot_confusion_matrix(y_true, y_pred, labels, output_file='confusion_matrix.png'):
    """
    Generate and save a confusion matrix visualization.
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        labels: List of unique label names
        output_file: Name of the output PNG file
    """
    # Flatten the lists of lists
    y_true_flat = [label for sent in y_true for label in sent]
    y_pred_flat = [label for sent in y_pred for label in sent]
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)
    
    # Convert to DataFrame for better labeling
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    # Create figure with larger size
    plt.figure(figsize=(15, 12))
    
    # Create a mask for the diagonal
    mask = np.zeros_like(cm_df, dtype=bool)
    np.fill_diagonal(mask, True)
    
    # Create two heatmaps: one for diagonal, one for off-diagonal
    # First, plot the off-diagonal elements
    sns.heatmap(cm_df, mask=mask, annot=True, fmt='d', cmap='Blues',
                xticklabels=True, yticklabels=True, cbar=False,
                vmax=cm_df.values.max() * 0.3)  # Limit color scale for off-diagonal
    
    # Then, plot the diagonal elements with a different color scale
    sns.heatmap(cm_df, mask=~mask, annot=True, fmt='d', cmap='Reds',
                xticklabels=True, yticklabels=True, cbar=False,
                vmin=0, vmax=cm_df.values.max())
    
    # Add a colorbar
    plt.colorbar(plt.gca().collections[0], label='Count')
    
    # Customize the plot
    plt.title('Confusion Matrix for NER Model', pad=20, fontsize=14)
    plt.xlabel('Predicted Labels', labelpad=10)
    plt.ylabel('True Labels', labelpad=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nConfusion matrix saved as {output_file}")

def main():
    # 1. Load training data
    print("\n=== Loading Training Data ===")
    train_sentences = load_all_training_data("training_data")
    
    if not train_sentences:
        print("Error: No training data loaded!")
        return
    
    # 2. Load testing data
    print("\n=== Loading Testing Data ===")
    test_sentences = load_data_from_csv("testing_data.csv")
    
    if not test_sentences:
        print("Error: No testing data loaded!")
        return
    
    # 3. Prepare features and labels
    print("\n=== Preparing Features and Labels ===")
    X_train = [sent2features(s) for s in train_sentences]
    y_train = [sent2labels(s) for s in train_sentences]
    
    X_test = [sent2features(s) for s in test_sentences]
    y_test = [sent2labels(s) for s in test_sentences]
    
    print(f"Number of training sentences: {len(X_train)}")
    print(f"Number of testing sentences: {len(X_test)}")
    
    # 4. Train the model
    print("\n=== Training Model ===")
    crf_model = train_model(X_train, y_train)
    
    # 5. Evaluate the model
    print("\n=== Evaluating Model ===")
    y_pred = crf_model.predict(X_test)
    
    # Get unique labels
    labels = sorted(list(set([label for sent in y_test for label in sent])))
    
    # Generate confusion matrix
    print("\n=== Generating Confusion Matrix ===")
    plot_confusion_matrix(y_test, y_pred, labels)
    
    # Print evaluation metrics
    evaluate_model(crf_model, X_test, y_test)

if __name__ == "__main__":
    main() 