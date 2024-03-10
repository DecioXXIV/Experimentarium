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

    # Special tokens
        # [CLS]: in a nutshell, represents the whole sentence
        # [SEP]: indicates the end of a sentence
        # [PAD]: padding token -> used to fill the rest of a sequence of tokens if it's shorter than the max length (BERT wants sequences all of the same length)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    tokens = tokens + ['[PAD]'] * ( - len(tokens))

    # Each token is uniquely identified by its ID -> consequently, each ID will be mapped to a unique vector of shape (768,)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = torch.tensor(token_ids).unsqueeze(0).to(device)

    # Attention Mask: list of binary values. 
        # If "attention_mask[i] == 1", we are asking BERT to pay attention to the i-th token
        # Otherwise, if "attention_mask[i] == 0", BERT will ignore the i-th token
    attention_mask = [1 if i != ['[PAD]'] else 0 for i in tokens]
    attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(token_ids, attention_mask)
    vec = output[1].cpu().numpy().flatten()

    # output charactheristics:
        # output[0] = last_hidden_state -> tensor of shape (batch_size, relevant_tokens, latent_space_dimension): contains the vectors of the last hidden layer of the model for each token in the input sentence
        # output[1] = pooler_output -> tensor of shape (batch_size, latent_space_dimension): contains the vector representation of the whole sentence (conventionally, it's the vector representation of the [CLS] token)
    
    if device == 'cuda':
        del token_ids
        del attention_mask
        del output
        torch.cuda.empty_cache()
    
    return vec