
# Named Entity Recognition for Code-Mixed Text: Project Outline

## 1. Problem Statement

Named Entity Recognition (NER) in code-mixed text presents unique challenges compared to monolingual text. Code-mixing, where speakers alternate between multiple languages within a conversation or even within a sentence, is extremely common in multilingual communities, particularly in Indian social media contexts. This project focuses on developing an effective NER system specifically for Hindi-English code-mixed text that can accurately identify and classify named entities despite the complexities introduced by language mixing.

### Problem Niche

Standard NER systems struggle with code-mixed text due to:
- Language identification challenges (determining which parts are in which language)
- Non-standard spellings and transliterations
- Mixing of scripts (Roman, Devanagari)
- Culturally-specific entities that might not exist in monolingual datasets
- Lack of annotated code-mixed corpora for training

## 2. Project Scope

The project will encompass:

1. **Dataset Creation and Annotation**:
   - Collection of Hindi-English code-mixed social media text from platforms like Twitter, Facebook, and WhatsApp
   - Development of comprehensive annotation guidelines for named entities in code-mixed text
   - Creation of a gold-standard annotated corpus with at least 1000 sentences

2. **Entity Types**:
   - Person (PER)
   - Location (LOC)
   - Organization (ORG)
   - Date/Time (DATE)
   - Culturally-specific entities (e.g., festivals, local events) (CSE)
   - Products, brands (PROD)

3. **Model Development**:
   - Feature-based Conditional Random Field (CRF) model.
   <!-- - Deep learning approaches (BiLSTM-CRF with multilingual embeddings) -->
   <!-- - Transformer-based approaches (code-mixed BERT variants) -->

4. **Evaluation**:
   - Standard metrics: Precision, Recall, F1-score
   - Analysis across different entity types
   - Analysis based on code-mixing density and patterns

<!-- 5. **Demo Application**: -->
   <!-- - Web interface for real-time NER on code-mixed text input -->

## 3. Data Exploration and Creation

### Existing Datasets
Following are the datasets that we will be using in the project

1. **Replication Data for Automatic language identification in code-switched Hindi-English social media text**
   - link for the dataset: [link](https://dataverse.harvard.edu/dataset.xhtml;jsessionid=d99a0d881cd919d13cd21af594bf?persistentId=doi:10.7910/DVN/QD94F9)

2. **Code-Mixed Dataset for Hindi-English**: Contains code-mixed social media posts
   - Data is scraped manuallay from social media posts, Link to the dataset will be provided later.

<!-- 3. **LinCE Benchmark**: A benchmark for linguistic code-switching evaluation -->
   <!-- - Source: [LinCE](https://github.com/googleinterns/mixmatch) -->

### Data Creation Process

1. **Collection**:
   - Scrape tweets and posts  with Hindi-English hashtags and topics
   - Collect public Facebook posts and comments from Indian pages

2. **Filtering**:
   - Filter for sentences with substantial mixing (at least 20% of both languages)

3. **Annotation**:
   - Manually annotate the dataset extracted from social media.
   - Dataset taken from Harvard is already annotated. 

### Annotation Guidelines

Detailed guidelines will be provided:
- Entity boundaries in mixed-language contexts
- Handling transliteration variations
- Culturally-specific entities classification
- Script variation handling

## 4. Technical Approach

### A. Preprocessing

1. **Text Normalization**:
   - Handling of non-standard spellings
   - Transliteration normalization using tools like [indic-trans](https://github.com/libindic/indic-trans)

2. **Language Identification**:
   - Word-level language tagging using models like [MultiLID](https://github.com/clab/fast_align)
   - Script identification (Roman vs. Devanagari)

### B. Feature Engineering for CRF

1. **Lexical Features**:
   - Word identity, prefix/suffix n-grams
   - Word shape (capitalization, digits, etc.)
   - Language tag of word

2. **Contextual Features**:
   - Previous/next words and their language tags
   - Word n-grams

3. **Dictionary Features**:
   - Gazetteers for both Hindi and English entities
   - Transliteration dictionaries

4. **Syntactic Features**:
   - POS tags from code-mixed POS taggers
   - Chunking information if available

### C. Model Development

1. **CRF Implementation**:
   - Using [sklearn-crfsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite) or [CRF++](https://taku910.github.io/crfpp/)
   - Feature selection using grid search and cross-validation

2. **Deep Learning Approach**:
   - BiLSTM-CRF with code-mixed word embeddings
   - Using [Glot500](https://github.com/microsoft/Glot500) or [MBERT](https://github.com/google-research/bert/blob/master/multilingual.md) for representations

3. **Transformer-based Models**:
   - Fine-tuning models like [MuRIL](https://tfhub.dev/google/MuRIL/1) (Multilingual Representations for Indian Languages)
   - Code-mixed BERT variants using [HuggingFace Transformers](https://github.com/huggingface/transformers)

## 5. Literature Review

Several research papers provide valuable insights for this project:

1. **[A Dataset for Named Entity Recognition in Indian Languages](https://www.aclweb.org/anthology/L16-1689.pdf) by Murthy et al. (2016)**:
   - Created a multi-lingual NER dataset for Indian languages
   - Provides insights on annotation challenges

2. **[Code Mixed Entity Extraction in Indian Languages Using Neural Networks](https://arxiv.org/abs/1908.09681) by Priyadharshini et al. (2019)**:
   - Uses BiLSTM-CRF for code-mixed Tamil-English
   - Shows improvements over traditional CRF approaches

3. **[Sentiment Analysis of Code-Mixed Indian Social Media Text](https://www.aclweb.org/anthology/W18-6120.pdf) by Patra et al. (2018)**:
   - While focused on sentiment analysis, provides insights on handling code-mixed text

4. **[CALCS 2018 Shared Task: Named Entity Recognition on Code-switched Data](https://www.aclweb.org/anthology/W18-3219.pdf) by Aguilar et al. (2018)**:
   - Describes methods and results from a shared task on NER for code-switched data
   - Provides evaluation metrics and baseline approaches

## 6. Implementation Plan

### Phase 1: Data Collection and Annotation
- Develop annotation guidelines
- Collect raw code-mixed data
- Train annotators and begin annotation process
- Achieve target corpus size with high inter-annotator agreement

### Phase 2: Model Development
- Implement preprocessing pipeline
- Develop and train CRF models
- Experiment with deep learning approaches
- Compare and optimize models

### Phase 3: Evaluation and Analysis
- Comprehensive evaluation using held-out test set
- Error analysis and model refinement
- Performance analysis across entity types and code-mixing patterns

### Phase 4: Demo Development 
- Build web interface for demonstration
- Integrate model into API
- User testing and refinement

## 7. Evaluation Metrics

1. **Standard NER Metrics**:
   - Precision, Recall, and F1-score for each entity type
   - Overall micro and macro F1 scores

2. **Code-Mixing Specific Analysis**:
   - Performance as a function of code-mixing index (CMI)
   - Analysis based on script variation patterns
   - Performance on mixed-language entities vs. single-language entities

## 8. Required Resources and Tools

### Libraries and Frameworks:
- [spaCy](https://spacy.io/) for NLP pipeline components
- [sklearn-crfsuite](https://sklearn-crfsuite.readthedocs.io/) for CRF implementation
- [PyTorch](https://pytorch.org/) for deep learning models
- [HuggingFace Transformers](https://huggingface.co/transformers/) for transformer models
- [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library) for Indian language processing

### Computing Resources:
- GPU-equipped machine for deep learning model training
- Cloud storage for dataset management

### Annotation Tools:
- [Doccano](https://github.com/doccano/doccano) for collaborative annotation
- [BRAT](https://brat.nlplab.org/) as an alternative annotation tool

## 9. Potential Challenges and Solutions

1. **Data Sparsity**:
   - Solution: Data augmentation techniques, semi-supervised learning approaches

2. **Annotation Consistency**:
   - Solution: Detailed guidelines, regular annotator meetings, adjudication process

3. **Handling Transliteration Variations**:
   - Solution: Normalization rules, character-level models, phonetic matching

4. **Model Performance on Low-Resource Languages**:
   - Solution: Transfer learning from high-resource languages, multilingual representations

## 10. Conclusion

This project will create a valuable resource and system for NER in Hindi-English code-mixed text, addressing a significant gap in current NLP capabilities for Indian languages. The developed methods can potentially be extended to other Indian language pairs and can inform approaches for other code-mixed NLP tasks. The annotated corpus will also serve as a benchmark dataset for future research in this area.

Human Language Technologies Research Lab at IIIT-Hyderabad, LCS2 at IIT-Madras, and Microsoft Research India have all conducted significant work in code-mixed NLP for Indian languages, and their methodologies and resources will be valuable references for this project.
