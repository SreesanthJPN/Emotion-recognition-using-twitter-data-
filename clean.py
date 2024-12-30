import re
import string
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def filter(data):
    username_pattern = r'^@'
    hashtag_pattern = r'^#'
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    filtered_data = []

    for i in data:
        if not isinstance(i, str):
            i = str(i)
        split = i.split(' ')
        
        split = [k for k in split if not re.match(username_pattern, k)]
        
        split = [k[1:] if re.match(hashtag_pattern, k) else k for k in split]
        
        split = [''.join(char for char in k if char not in string.punctuation) for k in split]
        
        tokens = word_tokenize(' '.join(split))
        
        processed_tokens = [
            stemmer.stem(word) for word in tokens 
            if word.lower() not in stop_words and not word.isnumeric()
        ]
        
        filtered_tweet = ' '.join(processed_tokens)
        filtered_data.append(filtered_tweet)

    return filtered_data


def create_labels(labels):
    label_mapping = {"positive": 0, "negative": 1, "neutral": 2}
    return labels.map(lambda x: label_mapping[x])