import nltk
from datetime import datetime
from collections import defaultdict
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import RegexpTokenizer

### ################## ###
### PRINCIPAL FUNCTION ###
### ################## ###

def prepare_dataset(dataset, lemmatization, stopw_rem):
    print("\nBeginning the Cleaning process on the Dataset...")
    nltk.download('stopwords')
    nltk.download('wordnet')

    cleaned_texts = list()
    texts = dataset['text'].tolist()

    regexp_tokenizer = RegexpTokenizer(r'\w+')
    
    tag_map = defaultdict(lambda : wordnet.NOUN)
    tag_map['J'] = wordnet.ADJ
    tag_map['V'] = wordnet.VERB
    tag_map['R'] = wordnet.ADV
    lemmatizer = WordNetLemmatizer()

    start = datetime.now()

    for t in texts:
        # Remove all the punctuation characters from the text instance
        tokens = remove_punctuation(t, regexp_tokenizer)
        
        if lemmatization is True:
            # Lemmatization of the tokens
            tokens = lemmatize_instance(tokens, tag_map, lemmatizer)
        
        if stopw_rem is True:
            # Remove all the stopwords from the lemmatized tokens
            tokens = remove_stopwords(tokens)
        
        cleaned_texts.append(' '.join(tokens))
    
    end = datetime.now()
    print("Cleaning process completed in:", end - start)

    return cleaned_texts

### ################# ###
### SUPPORT FUNCTIONS ###
### ################# ###

def remove_punctuation(instance, regexp_tokenizer):
    filtered_tokens = [t.lower() for t in regexp_tokenizer.tokenize(instance) if t.isalnum()]
    return filtered_tokens

def lemmatize_instance(tokens, tag_map, lemmatizer):
    lem_tokens = list()

    for token, tag in pos_tag(tokens):
        lemma = lemmatizer.lemmatize(token, tag_map[tag[0]])
        lem_tokens.append(lemma)
    
    return lem_tokens

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = list()

    for token in tokens:
        if token not in stop_words:
            filtered_tokens.append(token)
    
    return filtered_tokens