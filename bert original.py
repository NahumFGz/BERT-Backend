# # # # # # # # # # 
#    KT - BERT    #
# # # # # # # # # # 
import os
import re
import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from textwrap import wrap
from transformers import BertModel
import torch.nn.functional as F
from transformers import BertTokenizer

device = torch.device("cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

#PATH_SAVE_MODEL_1 = "model1.pth"
PATH_SAVE_MODEL_1 = "Twitter_model1.pth"

max_len       = 300          # limitaciones porque google collab tiene un ram limitado
batch_size    = 16           # Paquetes de 16 elementos
nclases       = 2            # Comentarios positivos y negativos
#num_epochs    = 2
#learning_rate = 5e-5 


# Create the BertClassfier class
class BertClassifier(nn.Module):
    
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, nclases

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-cased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits

# Load model
bert_classifier = torch.load(PATH_SAVE_MODEL_1, map_location='cpu')

# Create a function to tokenize a set of texts
def preprocessing_for_bert(data, tokenizer, max_len):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []
    
    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=sent,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=max_len,             # Max length to truncate/pad
            padding="max_length",           # Pad sentence to max length, # pad_to_max_length=True
            truncation=True,
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
        )
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
 
    return input_ids, attention_masks

# For individual use
def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    #print("probs", probs)
    #print()
    return probs

def classifier_treatment_load(text):
    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    labels = ["Violence","No Violence"]
    text_pd = pd.Series(text)
    test_inputs, test_masks = preprocessing_for_bert(
        text_pd, tokenizer, max_len)

    # Create the DataLoader for our test set
    test_dataset = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=batch_size)
    
    probs = bert_predict(bert_classifier, test_dataloader)
    preds = probs[:, 1]

    # Get accuracy over the test set
    id_y_pred = probs.argmax(axis=1)
    print("\n".join(wrap(text)))

    #print(f"Expert: {label}")
    print(f"Model : {labels[id_y_pred[0]]} - {probs[0][id_y_pred[0]]:.3f}")
    return {
        "text": text,
        "label": labels[id_y_pred[0]],
        "prob": float(probs[0][id_y_pred[0]])
    }


# # # # # # # # # # # # 
#    API KT - BERT    #
# # # # # # # # # # # # 

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Bienvenido a API KT-BERT"


@app.route("/predict", methods=['POST'])
def predict():
    json = request.get_json(force=True)
    print('----------------------> ',json)
    print(type(json))
    print(json['text'])

    text = json['text']
    result = classifier_treatment_load(text)
    return result


if __name__ == '__main__':
    app.run(port=3000, debug=True)

