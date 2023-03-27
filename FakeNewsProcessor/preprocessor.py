import pandas as pd
import numpy as np
import os

# Libraries requied for cleaning dataset
import nltk
import re
import string

# To remove stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

# for root word
from nltk.corpus import wordnet
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')


def text_cleaner(text):
    # Make text lower case
    text = text.lower()
    
    # Remove brackets
    text = re.sub('\[.*?\]', '', text)
    
    # Remove profile / resource links and website links
    text = re.sub('https?://\S+|www\.\S+', '', text)
    
    # Remove digits (with space)
    text = re.sub(r'\d+', '', text)
    
    # Remove HTML tags
    text = re.sub('<.*?>+', '', text)
    
    # Remove puncuation (with space)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    
    # Remove extra space
    text = re.sub('\s+', ' ', text)
    
    return text


def stopword_remover(text):
    
    # Set lanuage to english
    stop_words = set(stopwords.words('english'))
    
    # Tokenise the words
    token_words = word_tokenize(text)
    
    # Consider words which are NOT stopwords
    cleaned_words = [w for w in token_words if not w.lower() in stop_words]
    
    # Un-tokenise the words
    text_cleaned = " ".join(cleaned_words)  
    
    return text_cleaned


# Helper Function to get the part of speech (POS) of the input word
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"V": wordnet.VERB,
                "J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def pos_lemmatizer(text):
    # Initiate the Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatized the text wrt to POS tag
    lemma_words = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)]
    
    # Join all words in the text with space between them (forming sentence)    
    lemma_text = " ".join(lemma_words)
    
    # Return lemmatized string
    return lemma_text


def preprocess(data, col_name):
    
    # Cleaning dataset
    data['cleaned_' + col_name] = data[col_name].apply(text_cleaner)
    print('Data cleaned')
    
    # remove stopwords from dataset
    data['cleaned_' + col_name] = data['cleaned_' + col_name].apply(stopword_remover)
    print('Data stopwords removed')
    
    # lemmatize text in dataset
    data['cleaned_' + col_name] = data['cleaned_' + col_name].apply(pos_lemmatizer)
    print('Data lemmatized')

    return data


def senti_word_count(text, word_list):
    senti_words = [x for x in nltk.word_tokenize(text) if x in word_list]
    return senti_words

def text_features(data):
    senti_words = pd.read_excel(os.path.join(os.getcwd(), 'Resources', "Positive and Negative Word List.xlsx"))
    neg_words, pos_words = list(senti_words['Negative Sense Word List']),list(senti_words['Positive Sense Word List'].dropna())

    data['sent_length'] = data['cleaned_text'].apply(lambda x: len(x))
    data['word_count'] = data['cleaned_text'].apply(lambda x: len(x.split()))
    data['pos_count'] = data['cleaned_text'].apply(lambda x: len(senti_word_count(x, pos_words)))
    data['neg_count'] = data['cleaned_text'].apply(lambda x: len(senti_word_count(x, neg_words)))
    
    return data