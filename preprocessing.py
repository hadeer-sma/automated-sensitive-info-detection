from nltk.tag import StanfordNERTagger
import nltk
import en_core_web_sm
from textblob import TextBlob
import re
from nltk.corpus import stopwords
import pandas as pd

# Regular expressions for text cleaning
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

# Load English language model for named entity recognition
nlp = en_core_web_sm.load()

# Initialize Stanford NER tagger
st = StanfordNERTagger('stanford-ner-4.0.0/stanford-ner-4.0.0/classifiers/english.all.3class.distsim.crf.ser.gz',
                       'stanford-ner-4.0.0/stanford-ner-4.0.0/stanford-ner.jar', encoding='utf-8')

# Function to check if a location exists in text
def check_location(text):
    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text))):
        if hasattr(chunk, "label"):
            if chunk.label() == "GPE" or chunk.label() == "GSP":
                return "True"
    return "False"

# Functions to check different features in text
def check_PERSON(text):
    doc = nlp(text)
    labels = [X.label_ for X in doc.ents]
    return 'PERSON' in labels

def check_LOC(text):
    doc = nlp(text)
    labels = [X.label_ for X in doc.ents]
    return 'LOC' in labels

def check_Date(text):
    doc = nlp(text)
    labels = [X.label_ for X in doc.ents]
    return 'Date' in labels

def check_GPE(text):
    doc = nlp(text)
    labels = [X.label_ for X in doc.ents]
    return 'GPE' in labels

def check_TIME(text):
    doc = nlp(text)
    labels = [X.label_ for X in doc.ents]
    return 'TIME' in labels

def check_Place(text):
    words = TextBlob(text).words.lower()
    List1 = ['beach', 'coast', 'hotel', 'conference', 'island', 'airport', 'flight']
    return any(item in List1 for item in words)

def check_Postive(text):
    words = TextBlob(text).words.lower()
    List1 = ['go', 'going', 'leave', 'leaving', 'pack', 'booked', 'before', 'will', 'until', 'wait', 'plan',
             'ready', 'here', 'come', 'looking', 'forward']
    return any(item in List1 for item in words)

def check_Negative(text):
    words = TextBlob(text).words.lower()
    List1 = ['need', 'wish', 'not', 'no', 'want', 'wanna', 'back', 'went', 'may', 'might', 'maybe', 'had', 'recent',
             'was', 'were', 'could', 'should', 'hope', 'got', 'suppose', 'if', "didn't"]
    return any(item in List1 for item in words)

def check_tag(text):
    return "#" in text

def check_url(text):
    return "http" in text

def get_features(text):
    return [
        check_PERSON(text),
        check_LOC(text),
        check_Date(text),
        check_GPE(text),
        check_TIME(text),
        check_Place(text),
        check_Postive(text),
        check_Negative(text),
        check_url(text),
        check_tag(text)
    ]

    def clean_text(text):
        """
        text: a string
        return: modified initial string
        """
        text = text.lower()  # Lowercase text
        text = REPLACE_BY_SPACE_RE.sub(' ', text)  # Replace symbols with space
        text = BAD_SYMBOLS_RE.sub(' ', text)  # Remove symbols
        text = text.replace('x', ' ')  # Replace 'x' with space
        text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # Remove stopwords
        return text

    # Load the data
    data = pd.DataFrame()
    data = pd.read_csv("new_test.csv")

    # Extract features from text
    data['Person'] = data.text.apply(check_PERSON)
    data['Loc'] = data.text.apply(check_LOC)
    data['Date'] = data.text.apply(check_Date)
    data['Gpe'] = data.text.apply(check_GPE)
    data['Time'] = data.text.apply(check_TIME)
    data['Place'] = data.text.apply(check_Place)
    data['Pos'] = data.text.apply(check_Postive)
    data['Neg'] = data.text.apply(check_Negative)
    data['Url'] = data.text.apply(check_url)
    data['Tag'] = data.text.apply(check_tag)

    # Clean the text
    data['text'] = data.text.apply(clean_text)

    # Save the data with features
    data.to_csv("test_with_features.csv")

