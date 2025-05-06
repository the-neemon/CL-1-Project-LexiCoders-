import sklearn_crfsuite
from sklearn_crfsuite import metrics
import re
import csv
import pandas as pd
import os
import glob

# --- Feature Extraction ---

def word2features(sent, i):
    """Extract features for a word at position i in a sentence."""
    word = sent[i][0]
    # Placeholder for POS tag - replace with actual POS tagging if available
    postag = 'TAG' 
    # Placeholder for Language ID - replace with actual language identification
    lang = 'LANG' 

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag, # Add POS tag feature
        'lang': lang, # Add language tag feature
        # Add more features based on outline.md (e.g., word shape, prefixes)
        'word.shape': get_word_shape(word),
        'word[:1]': word[:1],
        'word[:2]': word[:2],
        'word[:3]': word[:3],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = 'TAG' # Placeholder
        lang1 = 'LANG' # Placeholder
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:lang': lang1,
        })
    else:
        features['BOS'] = True # Beginning of Sentence

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = 'TAG' # Placeholder
        lang1 = 'LANG' # Placeholder
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:lang': lang1,
        })
    else:
        features['EOS'] = True # End of Sentence

    # Add more contextual features (e.g., previous/next 2 words) if needed

    return features

def get_word_shape(word):
    """Get the 'shape' of a word (e.g., 'Xxxx' for 'Word')."""
    shape = re.sub(r'\w', 'x', word)
    shape = re.sub(r'\d', 'd', shape)
    return shape

def sent2features(sent):
    """Convert a sentence (list of (word, tag) tuples) to features."""
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    """Extract labels from a sentence (list of (word, tag) tuples)."""
    return [label for token, label in sent]

def sent2tokens(sent):
    """Extract tokens from a sentence (list of (word, tag) tuples)."""
    return [token for token, label in sent]

# --- Data Loading ---

def load_data_from_csv(filepath):
    """
    Load data from a CSV file with format:
    Sent,Word,Tag
    sent: 0,word1,B-Per
    sent: 0,word2,Other
    ...
    
    Returns a list of sentences, where each sentence is a list of (word, tag) tuples.
    """
    print(f"Loading data from CSV: {filepath}")
    sentences = []
    
    try:
        # Read CSV file with pandas
        df = pd.read_csv(filepath)
        # Ensure all words are treated as strings
        df['Word'] = df['Word'].astype(str)
        # Group by sentence ID
        for sent_id, group in df.groupby('Sent'):
            sentence = []
            for _, row in group.iterrows():
                word = str(row['Word'])  # Ensure word is a string
                tag = row['Tag']
                # Convert 'Other' tag to 'O' to match standard NER format
                if tag == 'Other':
                    tag = 'O'
                sentence.append((word, tag))
            sentences.append(sentence)
            
        print(f"Loaded {len(sentences)} sentences with a total of {sum(len(s) for s in sentences)} tokens")
        
        # Validate the data
        if sentences:
            print(f"First sentence sample: {sentences[0][:5]}...")
        else:
            print("Warning: No sentences were loaded!")
            
    except Exception as e:
        print(f"Error loading data from CSV {filepath}: {e}")
        return []
        
    return sentences

def load_test_data_from_file(filepath):
    """
    Load data from the testing_annotated_data file format.
    Each line contains a word and its tag separated by a space.
    Sentences are separated by blank lines.
    Returns a list of sentences, where each sentence is a list of (word, tag) tuples.
    """
    print(f"Loading test data from: {filepath}")
    sentences = []
    current_sentence = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines at the beginning
                if line.startswith('#') or line.startswith('//'):
                    continue
                    
                if not line:  # Blank line indicates end of sentence
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        word = parts[0]
                        tag = parts[1]
                        # Convert 'Other' tag to 'O' to match standard NER format
                        if tag == 'Other':
                            tag = 'O'
                        current_sentence.append((word, tag))
        
        # Add the last sentence if file doesn't end with a blank line
        if current_sentence:
            sentences.append(current_sentence)
            
        print(f"Loaded {len(sentences)} sentences with a total of {sum(len(s) for s in sentences)} tokens")
        
        # Validate the data
        if sentences:
            print(f"First sentence sample: {sentences[0][:5]}...")
        else:
            print("Warning: No sentences were loaded!")
            
    except Exception as e:
        print(f"Error loading test data from {filepath}: {e}")
        return []
        
    return sentences

def load_data(filepath):
    """
    Load data from an IOB formatted file.
    Each line: word\tTAG (or word TAG)
    Sentences separated by blank lines.
    Returns a list of sentences, where each sentence is a list of (word, tag) tuples.
    """
    print(f"Loading data from {filepath}")
    sentences = []
    current_sentence = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line: # Blank line indicates end of sentence
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                else:
                    parts = line.split() # Assumes space or tab separated
                    if len(parts) >= 2:
                        # Last part is the tag, everything else is the word (handles multi-word tokens)
                        word = " ".join(parts[:-1])
                        tag = parts[-1]
                        current_sentence.append((word, tag))
                    elif len(parts) == 1: # Handle cases with only word
                        print(f"Warning: Line {i} has no tag: '{line}'. Using 'O' as default.")
                        current_sentence.append((parts[0], 'O'))
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return []
        
    # Add the last sentence if the file doesn't end with a blank line
    if current_sentence:
        sentences.append(current_sentence)
        
    print(f"Loaded {len(sentences)} sentences with a total of {sum(len(s) for s in sentences)} tokens")
    
    # Validate the data
    if sentences:
        print(f"First sentence sample: {sentences[0][:5]}...")
    else:
        print("Warning: No sentences were loaded!")
        
    return sentences

def load_all_training_data(directory_path):
    """
    Load all training data from CSV files in the specified directory.
    Returns a list of sentences, where each sentence is a list of (word, tag) tuples.
    """
    print(f"Loading all training data from directory: {directory_path}")
    all_sentences = []
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    if not csv_files:
        print(f"Warning: No CSV files found in {directory_path}")
        return []
    
    print(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
    
    # Load data from each CSV file
    for csv_file in csv_files:
        print(f"Processing file: {os.path.basename(csv_file)}")
        sentences = load_data_from_csv(csv_file)
        all_sentences.extend(sentences)
    
    print(f"Loaded a total of {len(all_sentences)} sentences with {sum(len(s) for s in all_sentences)} tokens from all files")
    return all_sentences

# --- Model Training ---

def train_model(X_train, y_train, model_path="crf_model.joblib"):
    """Train the CRF model and save it."""
    print("Starting model training...")
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    print("Model training complete.")

    # Save the model (optional, requires joblib)
    # import joblib
    # joblib.dump(crf, model_path)
    # print(f"Model saved to {model_path}")

    return crf

# --- Prediction ---

def predict(model, sentences_features):
    """Predict labels for new sentences."""
    return model.predict(sentences_features)

# --- Evaluation ---

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print classification report."""
    if not X_test or not y_test:
        print("Warning: Empty test data. Cannot evaluate model.")
        return
    
    y_pred = model.predict(X_test)

    # Ensure all labels are strings for metrics calculation
    flat_y_test = [label for sent in y_test for label in sent]
    flat_y_pred = [label for sent in y_pred for label in sent]

    # Get unique labels from both test data and model
    unique_labels = set()
    
    # Add non-O labels from test data
    for label in flat_y_test:
        if label != 'O':
            unique_labels.add(label)
    
    # Add non-O labels from predictions
    for label in flat_y_pred:
        if label != 'O':
            unique_labels.add(label)
    
    # Add any labels the model knows about
    if hasattr(model, 'classes_'):
        for label in model.classes_:
            if label != 'O':
                unique_labels.add(label)
    
    if not unique_labels:
        print("No entity labels found in test data or predictions.")
        print("This could be due to formatting issues or lack of labeled entities.")
        
        # Show sample of test data to help debug
        print("\nSample of test data:")
        for i, sentence in enumerate(y_test):
            if i >= 2:  # Only show first 2 sentences
                break
            print(f"Test sentence {i+1}: {sentence}")
        return
        
    # For reporting, map 'O' to 'Other' in y_test, y_pred, and labels
    def map_O_to_Other(seq):
        return [['Other' if label == 'O' else label for label in sent] for sent in seq]
    y_test_report = map_O_to_Other(y_test)
    y_pred_report = map_O_to_Other(y_pred)

    # Build label list for report, replacing 'O' with 'Other'
    labels = sorted(list(unique_labels))
    if 'O' in labels:
        labels = [l for l in labels if l != 'O'] + ['Other']
    elif 'Other' not in labels:
        labels.append('Other')

    # Filter out labels with zero support
    def get_label_support(y_true, y_pred, label):
        true_count = sum(1 for sent in y_true for tag in sent if tag == label)
        pred_count = sum(1 for sent in y_pred for tag in sent if tag == label)
        return true_count + pred_count

    # Keep only labels that have non-zero support
    labels = [label for label in labels if get_label_support(y_test_report, y_pred_report, label) > 0]
    print(f"Found entity labels: {labels}")
    
    try:
        print("Classification Report:")
        report = metrics.flat_classification_report(
            y_test_report, y_pred_report, labels=labels, digits=3, zero_division=0
        )
        print(report)
    except Exception as e:
        print(f"Error generating classification report: {e}")
    
    # Show sample predictions
    print("\nSample predictions:")
    for i, (test_sent, pred_sent) in enumerate(zip(y_test, y_pred)):
        if i >= min(3, len(y_test)):  # Only show first 3 sentences or less
            break
        print(f"\nSentence {i+1}:")
        
        # Extract words from features
        words = []
        for feat_dict in X_test[i]:
            for feat_name, feat_val in feat_dict.items():
                if feat_name == 'word.lower()':
                    words.append(feat_val)
                    break
            else:
                words.append("[UNKNOWN]")
        
        # Print word-label pairs
        for j, (word, true_label, pred_label) in enumerate(zip(words, test_sent, pred_sent)):
            match = "✓" if true_label == pred_label else "✗"
            print(f"  {word}: True={true_label}, Pred={pred_label} {match}")
    
    # Calculate overall metrics
    correct = sum(1 for true, pred in zip(flat_y_test, flat_y_pred) if true == pred)
    total = len(flat_y_test)
    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall token accuracy: {accuracy:.3f} ({correct}/{total})")

# --- Main Execution Example ---

if __name__ == "__main__":
    # 1. Load Annotated Data
    print("\n=== Loading annotated data ===")
    
    # Load all training data from the training_data directory
    train_sents = load_all_training_data('training_data')
    
    # If directory loading fails, try individual files as fallback
    if not train_sents:
        print("Failed to load from training_data directory. Trying individual files...")
        train_sents = load_data('annotated_train.txt')
        
        # If still no data, use dummy data for demonstration
        if not train_sents:
            print("\nUsing dummy data for demonstration as actual data files are empty or improperly formatted.")
            train_sents = [
                [('Yaar', 'O'), (',', 'O'), ('kal', 'B-DATE'), ('ka', 'O'), ('plan', 'O'), ('set', 'O'), ('hai', 'O'), ('na', 'O'), ('?', 'O')],
                [('Bro', 'O'), (',', 'O'), ('movie', 'B-PROD'), ('dekhne', 'O'), ('chalein', 'O'), ('kya', 'O'), ('?', 'O')]
            ]
    
    # Load test data from testing_annotated_data file
    test_sents = load_test_data_from_file('testing_annotated_data')
    
    # If test file loading fails, try alternative formats or use dummy data
    if not test_sents:
        print("Failed to load testing_annotated_data. Trying alternative formats...")
        test_sents = load_data('annotated_test.txt')
        
        # If still no data, use dummy data for demonstration
        if not test_sents:
            print("\nUsing dummy test data for demonstration as test files are empty or improperly formatted.")
            test_sents = [
                [('Seriously', 'O'), (',', 'O'), ('yeh', 'O'), ('new', 'O'), ('song', 'B-PROD'), ('super', 'O'), ('catchy', 'O'), ('hai', 'O'), ('.', 'O')],
                [('Guys', 'O'), (',', 'O'), ('kal', 'B-DATE'), ('ke', 'O'), ('event', 'B-CSE'), ('ke', 'O'), ('liye', 'O'), ('dress', 'O'), ('code', 'O'), ('kya', 'O'), ('hai', 'O'), ('?', 'O')]
            ]

    # 2. Prepare Data for CRF
    print("\n=== Preparing data for training ===")
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    
    print(f"Training data: {len(X_train)} sentences, {sum(len(x) for x in X_train)} tokens")
    print(f"Entity labels in training data: {sorted(set(label for sent in y_train for label in sent if label != 'O'))}")

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]
    
    print(f"Test data: {len(X_test)} sentences, {sum(len(x) for x in X_test)} tokens")
    print(f"Entity labels in test data: {sorted(set(label for sent in y_test for label in sent if label != 'O'))}")

    # 3. Train the Model
    crf_model = train_model(X_train, y_train)

    # 4. Evaluate the Model
    print("\n=== Evaluating on test data ===")
    evaluate_model(crf_model, X_test, y_test)

    # 5. Predict on Sample Sentences
    print("\n=== Predicting on new sample sentences ===")
    sample_sentences_raw = [
        "Rahul Sharma Delhi jaayega next week",
        "Diwali ko main Mumbai mein celebrate karunga",
        "Google ne new phone launch kiya hai"
    ]
    print(f"Sample sentences for testing: {sample_sentences_raw}")

    # Basic tokenization and feature extraction for sample sentences
    sample_sentences_tokenized = [[(word, 'O') for word in sent.split()] for sent in sample_sentences_raw]
    X_sample = [sent2features(s) for s in sample_sentences_tokenized]
    
    try:
        # Make predictions
        y_sample_pred = predict(crf_model, X_sample)
        
        # Display results
        print("\nPrediction results:")
        for i, sent_raw in enumerate(sample_sentences_raw):
            print(f"\nSentence: {sent_raw}")
            tokens = [tok[0] for tok in sample_sentences_tokenized[i]]
            predictions = y_sample_pred[i]
            
            # Format and display predictions
            formatted_results = []
            for token, pred_label in zip(tokens, predictions):
                if pred_label != 'O':
                    formatted_results.append(f"{token}[{pred_label}]")
                else:
                    formatted_results.append(token)
                    
            print(f"Parsed: {' '.join(formatted_results)}")
            print(f"Entities: {[f'{tokens[j]}({pred})' for j, pred in enumerate(predictions) if pred != 'O']}")
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Please check that the model was trained successfully and that the sample sentences are properly formatted.")

