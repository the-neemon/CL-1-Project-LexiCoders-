from model import (
    load_all_training_data,
    load_data_from_csv,
    sent2features,
    sent2labels,
    train_model,
    evaluate_model
)

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
    evaluate_model(crf_model, X_test, y_test)

if __name__ == "__main__":
    main() 