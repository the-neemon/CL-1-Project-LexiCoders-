import sklearn_crfsuite
from sklearn_crfsuite import metrics
import re

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
        
    labels = sorted(list(unique_labels))
    print(f"Found entity labels: {labels}")
    
    try:
        print("Classification Report:")
        report = metrics.flat_classification_report(
            y_test, y_pred, labels=labels, digits=3, zero_division=0
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
    # First try to load the actual annotated data files
    print("\n=== Loading annotated data ===")
    train_sents = load_data('annotated_train.txt')
    test_sents = load_data('annotated_test.txt')
    
    # If the actual files are empty, use dummy data for demonstration
    if not train_sents or not test_sents:
        print("\nUsing dummy data for demonstration as actual data files are empty or improperly formatted.")
        train_sents = [
            [('Yaar', 'O'), (',', 'O'), ('kal', 'B-DATE'), ('ka', 'O'), ('plan', 'O'), ('set', 'O'), ('hai', 'O'), ('na', 'O'), ('?', 'O')],
            [('Bro', 'O'), (',', 'O'), ('movie', 'B-PROD'), ('dekhne', 'O'), ('chalein', 'O'), ('kya', 'O'), ('?', 'O')]
        ]
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

