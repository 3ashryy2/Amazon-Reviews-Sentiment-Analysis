# Amazon Reviews Sentiment Analysis

This project aims to classify Amazon product reviews into three sentiment categories: Negative, Neutral, and Positive. The dataset consists of over 17,000 records. The project involves data preprocessing, word embedding, and training two models (SimpleRNN and LSTM) to predict the sentiment of the reviews. Additionally, a report is generated to determine the best hyperparameters for each model.

## Requirements

- Python 3.11
- pandas
- numpy
- scikit-learn
- keras
- nltk
- tqdm

## Dataset

The dataset `amazon_reviews.csv` contains the following columns:
- `sentiments`: The sentiment of the review (`neutral`, `positive`, `negative`).
- `cleaned_review`: The text of the review after preprocessing.
- `cleaned_review_length`: The number of words in the cleaned review.
- `review_score`: The rating given by the user.

## Data Pre-processing

The data preprocessing includes:
1. Loading the dataset.
2. Removing stopwords and non-alphabetic characters using NLTK.
3. Tokenizing and padding sequences for the reviews.

## Data Splitting

The data is split into training and validation sets with different splitting ratios (70%-30% and 80%-20%).

## Word Embedding

The word embedding process includes:
1. Building a vocabulary by extracting and indexing unique words from the reviews.
2. Converting each review into a sequence of indices.
3. Applying sequence padding to have all sequences of the same length.

## Model Training

Two models are trained using the preprocessed data:
1. SimpleRNN
2. LSTM

Dropout layers are added between dense layers to prevent overfitting. The models are trained and evaluated for different hyperparameters (splitting ratio and sequence padding length).

## Results

The results are saved and analyzed to determine the best hyperparameters for each model. The accuracy of each model is printed and the best model is identified.

## How to Run

1. Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn keras nltk tqdm
    ```

2. Download the dataset `amazon_reviews.csv` and place it in the project directory.

3. Open the jython notebook `amazon_analysis.ipynb` and run all cells.

