# Text Sentiment Classifier

## Overview
The Text Sentiment Classifier is a PyTorch-based project that utilizes a pre-trained BERT model to classify text into three sentiment categories: positive, negative, and neutral. It is designed to fine-tune the BERT model on labeled text data and predict the sentiment of new text inputs.

## Features
- Fine-tunes a pre-trained BERT model (`bert-base-uncased`) for sentiment classification.
- Implements data preprocessing, including cleaning and stemming.
- Custom PyTorch `Dataset` class to handle tokenized text inputs.
- Model training, evaluation, and prediction functionalities.
- Save and load the trained model for future use.

## Requirements
- Python 3.7+
- PyTorch
- Transformers
- NLTK
- scikit-learn
- pandas
- CUDA-enabled GPU (optional, but recommended for training)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install the required Python libraries:
   ```bash
   pip install torch transformers nltk scikit-learn pandas
   ```

3. Download the NLTK stopwords dataset:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

## Project Structure
```
.
├── model.py          # Defines the TextClassifier class.
├── Loader.py         # Defines the TextDataLoader class.
├── clean.py          # Contains text preprocessing and label creation functions.
├── train.py          # Training and evaluation script.
├── predict.py        # Prediction script for new text inputs.
├── data.csv          # Input dataset containing text and labels.
└── text_classifier.pt # Saved trained model (after training).
```

## Usage

### Data Preprocessing
1. Prepare your dataset in CSV format with columns:
   - `Caption`: Text data (tweets, sentences, etc.).
   - `LABEL`: Sentiment labels (positive, negative, neutral).

2. Run the `filter` and `create_labels` functions from `clean.py` to preprocess text and encode labels:
   ```python
   from clean import filter, create_labels

   data = pd.read_csv('data.csv')
   tweets = data['Caption']
   labels = data['LABEL']

   tweets = filter(tweets)
   labels = create_labels(labels)
   ```

### Training
1. Modify the `train.py` script with your dataset and model parameters.
2. Run the training script:
   ```bash
   python train.py
   ```
3. The trained model will be saved as `text_classifier.pt`.

### Prediction
1. Use the `predict.py` script to classify new text inputs:
   ```python
   from predict import predict

   test_texts = [
       "The product is amazing!",
       "I hate this experience.",
       "It’s okay, not great but not bad either."
   ]

   predictions = predict(test_texts, tokenizer, model)
   print("Predictions:", predictions)
   ```

### Output
- Predictions will be integers corresponding to the sentiment labels:
  - `0`: Positive
  - `1`: Negative
  - `2`: Neutral

## Customization
- Modify the `filter` function in `clean.py` to include/exclude specific preprocessing steps.
- Adjust training parameters (e.g., `batch_size`, `learning_rate`, and `epochs`) in the training script.
- Experiment with different pre-trained models by changing `model_name` in `train.py` and `predict.py`.

## License
This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments
- Hugging Face's Transformers library for pre-trained language models.
- PyTorch for the deep learning framework.
- NLTK for text preprocessing.

---
Feel free to reach out with any questions or contributions!

