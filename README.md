# NER Model for Code-Mixed Text (Hindi-English)

## Project Overview
This project implements a Named Entity Recognition (NER) model for code-mixed text containing both Hindi and English languages. The model uses a Conditional Random Field (CRF) approach to identify various named entities in the text.

## Web Interface
Visit our [web interface](https://the-neemon.github.io/CL-1-Project-LexiCoders-/) to see the model's performance and try it out with your own text.

## Directory Structure
```
.
├── model.py                 # Core model implementation
├── run_model.py            # Script to run the model
├── index.html              # Web interface
├── RESULTS.md              # Detailed results and analysis
├── error.md                # Error analysis
├── requirements.txt        # Python dependencies
├── testing_data.csv        # Test dataset
├── testing_annotated_data  # Annotated test data
├── confusion_matrix.png    # Generated confusion matrix
└── training_data/          # Training datasets
    ├── annotatedData.csv   # Main annotated dataset
    ├── naman_data_1.csv    # Naman's dataset 1
    ├── naman_data_2.csv    # Naman's dataset 2
    ├── naman_data_3.csv    # Naman's dataset 3
    ├── shrish_data.csv     # Shrish's dataset
    ├── train_datav1.csv    # Training data version 1
    ├── train_datav2.csv    # Training data version 2
    └── yash_data.csv       # Yash's dataset
```

## Entity Types
The model recognizes the following entity types:
- B-Cul/I-Cul: Cultural terms
- B-Kin/I-Kin: Kinship terms
- B-Loc/I-Loc: Locations
- B-Org/I-Org: Organizations
- B-Par/I-Par: Particles
- B-Per/I-Per: Person names
- B-Rel/I-Rel: Religious terms
- Other: Non-entity tokens

## Requirements
- Python 3.x
- Required packages:
  ```bash
  pip install sklearn_crfsuite pandas numpy matplotlib seaborn scikit-learn
  ```

## Instructions for Running the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/the-neemon/CL-1-Project-LexiCoders-.git
   cd CL-1-Project-LexiCoders-
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the model:
   ```bash
   python run_model.py
   ```
   This will:
   - Load all training data from the training_data directory
   - Train the CRF model
   - Evaluate the model on test data
   - Generate a confusion matrix (confusion_matrix.png)
   - Print evaluation metrics

## Model Performance
The model's performance metrics can be found in:
- `RESULTS.md`: Detailed performance metrics and analysis
- `error.md`: Error analysis and improvement suggestions
- `confusion_matrix.png`: Visual representation of model predictions

## Project Documentation
- `PROJECT_REPORT.pdf`: Comprehensive project report
- `LexiCoders-Outline.pdf`: Project outline and specifications
