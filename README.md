# Named Entity Recognition for Code-Mixed Text (Hindi-English)

This project implements a Named Entity Recognition (NER) model specifically designed for code-mixed text containing both Hindi and English languages. The model uses a Conditional Random Field (CRF) approach to identify various named entities in the text.

## Web Interface

The project has a web interface that showcases the model's performance, examples, and results. You can access it at:
[LexiCoders NER Web Interface](https://the-neemon.github.io/CL-1-Project-LexiCoders-/)

## Directory Structure

```
CL-1-Project-LexiCoders-/
├── training_data/              # Directory containing training datasets
│   ├── train_datav1.csv       # 100 sentences, 1,336 tokens
│   ├── shrish_data.csv        # 179 sentences, 1,624 tokens
│   ├── naman_data_2.csv       # 525 sentences, 7,144 tokens
│   ├── naman_data_3.csv       # 338 sentences, 3,209 tokens
│   ├── train_datav2.csv       # 100 sentences, 1,346 tokens
│   ├── annotatedData.csv      # 3,085 sentences, 68,506 tokens
│   ├── naman_data_1.csv       # 281 sentences, 4,976 tokens
│   └── yash_data.csv          # 241 sentences, 1,187 tokens
├── testing_data.csv           # Testing dataset (565 sentences, 11,662 tokens)
├── run_model.py              # Main script to run the model
├── model.py                  # Contains model implementation and helper functions
├── confusion_matrix.png      # Generated confusion matrix visualization
└── README.md                 # This file
```

## Entity Types Recognized

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
- Required Python packages:
  ```
  pip install sklearn_crfsuite
  pip install pandas
  pip install numpy
  pip install matplotlib
  pip install seaborn
  pip install scikit-learn
  ```

## Instructions for Running the Code

1. **Clone the Repository**
   ```bash
   git clone https://github.com/the-neemon/CL-1-Project-LexiCoders-.git
   cd CL-1-Project-LexiCoders-
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**
   - Ensure all training data files are in the `training_data` directory
   - Place the testing data file as `testing_data.csv` in the root directory

4. **Run the Model**
   ```bash
   python run_model.py
   ```
   This will:
   - Load all training data
   - Load testing data
   - Train the CRF model
   - Generate a confusion matrix visualization
   - Display evaluation metrics

5. **View Results**
   - The confusion matrix will be saved as `confusion_matrix.png`
   - Evaluation metrics will be displayed in the terminal
   - For a detailed analysis and examples, visit the [web interface](https://the-neemon.github.io/CL-1-Project-LexiCoders-/)

## Model Performance

The model achieves:
- Token Accuracy: 93.1%
- 10,859 correct predictions out of 11,662 tokens

Detailed entity-wise performance metrics can be found on the web interface.

## File Descriptions

- `run_model.py`: Main script that orchestrates the model training and evaluation process
- `model.py`: Contains the core model implementation, including:
  - Data loading functions
  - Feature extraction
  - CRF model training
  - Evaluation metrics
- `training_data/`: Directory containing all training datasets in CSV format
- `testing_data.csv`: The test dataset used for model evaluation
- `confusion_matrix.png`: Generated visualization of the confusion matrix

## Contributing

Feel free to submit issues and enhancement requests!
