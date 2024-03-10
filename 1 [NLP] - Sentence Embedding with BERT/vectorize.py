import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch import cuda
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer

### ################### ###
### PRINCIPAL FUNCTIONS ###
### ################### ###

def tfidf_vectorization(dataset):
    print("Starting the TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer().fit(dataset['text'])
    vectors = vectorizer.transform(dataset['text']).toarray()

    print("TF-IDF Vectorization completed successfully")
    print("Shape of the TF-IDF Vectors:", vectors.shape)
    print()

    return pd.DataFrame(vectors)

def bert_vectorization(dataset):
    print("Starting the BERT Feature Extraction...")
    if cuda.is_available():
        print("Device used to extract the Features with BERT:", cuda.get_device_name())
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        device = 'cuda'
    else:
        print("No GPU available. Using CPU")
        device = 'cpu'
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)

    texts = dataset['text'].tolist()
  
    vectors = np.zeros((len(dataset), 768))
    start = datetime.now()

    i = 0
    for t in texts:
        extraction = bert_extract_features(t, tokenizer, model, device)
        vectors[i] = extraction
        i += 1
    
    end = datetime.now()
    print("BERT Feature Extraction completed in:", end - start)
    print("Shape of the BERT Vectors:", vectors.shape)
    print()

    return pd.DataFrame(vectors)

### ################# ###
### SUPPORT FUNCTIONS ###
### ################# ###

def bert_extract_features(t, tokenizer, model, device, max_tokens=512):
    if len(t) > max_tokens:
        t = t[:max_tokens]

    tokens = tokenizer.tokenize(t)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    tokens = tokens + ['[PAD]'] * ( - len(tokens))

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = torch.tensor(token_ids).unsqueeze(0).to(device)

    attention_mask = [1 if i != ['[PAD]'] else 0 for i in tokens]
    attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(token_ids, attention_mask)
    vec = output[1].cpu().numpy().flatten()
    
    if device == 'cuda':
        del token_ids
        del attention_mask
        del output
        torch.cuda.empty_cache()
    
    return vec