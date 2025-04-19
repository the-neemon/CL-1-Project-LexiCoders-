# NER Model for Code-Mixed Text (Hindi-English) - Results and Documentation

## Project Overview
This project implements a Named Entity Recognition (NER) model for code-mixed text containing both Hindi and English languages. The model uses a Conditional Random Field (CRF) approach to identify various named entities in the text.

## Data Sources

### Training Data
The model was trained on 7 different datasets:
1. train_datav1.csv (100 sentences, 1,336 tokens)
2. shrish_data.csv (179 sentences, 1,624 tokens)
3. naman_data_2.csv (525 sentences, 7,144 tokens)
4. train_datav2.csv (100 sentences, 1,346 tokens)
5. annotatedData.csv (3,085 sentences, 68,506 tokens)
6. naman_data_1.csv (281 sentences, 4,976 tokens)
7. yash_data.csv (241 sentences, 1,187 tokens)

Total Training Data: 4,511 sentences with 86,119 tokens

### Testing Data
The model was tested on:
- 565 sentences with 11,662 tokens

## Model Performance

### Overall Performance
- Token Accuracy: 92.6% (10,803 correct out of 11,662 tokens)

### Entity-wise Performance

| Entity Type | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| B-Cul      | 0.562     | 0.146  | 0.232    | 123     |
| B-Kin      | 0.825     | 0.688  | 0.751    | 199     |
| B-Loc      | 0.882     | 0.724  | 0.795    | 290     |
| B-Org      | 0.753     | 0.534  | 0.625    | 343     |
| B-Par      | 0.943     | 0.827  | 0.881    | 139     |
| B-Per      | 0.763     | 0.659  | 0.707    | 132     |
| B-Rel      | 0.426     | 0.312  | 0.360    | 64      |
| I-Cul      | 0.714     | 0.118  | 0.202    | 85      |
| I-Kin      | 0.000     | 0.000  | 0.000    | 0       |
| I-Loc      | 0.855     | 0.405  | 0.550    | 116     |
| I-Org      | 0.748     | 0.554  | 0.637    | 166     |
| I-Par      | 0.000     | 0.000  | 0.000    | 0       |
| I-Per      | 0.730     | 0.740  | 0.735    | 73      |
| I-Rel      | 0.607     | 0.586  | 0.596    | 29      |
| Other      | 0.000     | 0.000  | 0.000    | 0       |

### Sample Predictions

#### Sentence 1:
```
aaj: True=O, Pred=O ✓
main: True=O, Pred=O ✓
andheri: True=B-Loc, Pred=O ✗
station: True=I-Loc, Pred=O ✗
pe: True=O, Pred=O ✓
late: True=O, Pred=O ✓
ho: True=O, Pred=O ✓
gaya: True=O, Pred=O ✓
yaar: True=B-Par, Pred=B-Par ✓
nan: True=O, Pred=O ✓
mumbai: True=B-Loc, Pred=B-Loc ✓
```

## Running the Model

### Prerequisites
- Python 3.x
- Required Python packages:
  - sklearn_crfsuite
  - pandas
  - numpy

### Steps to Run

1. **Data Preparation**
   - Ensure training data is in the `training_data` directory
   - Place testing data as `testing_annotated_data` in the root directory

2. **Train and Test the Model**
   ```bash
   python run_model.py
   ```
   This will:
   - Load all training data
   - Convert and load testing data
   - Train the model
   - Evaluate performance
   - Display results

## Project Assumptions

1. **Data Format**
   - Training data: CSV files with columns "Sent", "Word", "Tag"
   - Testing data: Space-separated format with word and tag per line
   - Sentences are separated by blank lines

2. **Entity Types**
   The model recognizes the following entity types:
   - B-Cul/I-Cul: Cultural terms
   - B-Kin/I-Kin: Kinship terms
   - B-Loc/I-Loc: Locations
   - B-Org/I-Org: Organizations
   - B-Par/I-Par: Particles
   - B-Per/I-Per: Person names
   - B-Rel/I-Rel: Religious terms
   - Other: Non-entity tokens

3. **Language Mix**
   - The model handles both Hindi and English text
   - No explicit language identification is performed
   - Features are language-agnostic

4. **Model Architecture**
   - Uses CRF for sequence labeling
   - Features include word-level and contextual information
   - No external resources (like word embeddings) are used

## Areas for Improvement

1. **Entity Recognition**
   - Improve performance on Cultural and Religious entities
   - Better handling of compound entities (e.g., "Dadar market")
   - Address missing support for some entity types (I-Kin, I-Par)

2. **Feature Engineering**
   - Add more sophisticated features for better entity recognition
   - Consider adding language-specific features
   - Explore using word embeddings

3. **Data Quality**
   - Address inconsistencies in entity labeling
   - Ensure balanced representation of all entity types
   - Add more training data for underrepresented entities 
