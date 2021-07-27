import re
import torch

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Replace '&amp' with '&'
    text = re.sub(r'&amp', '&', text)

    # Replace " with ''
    text = re.sub(r'"', '', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Replace HashTags
    text = re.sub(r'PorqueMeQuedé', 'Porque me quedé', text)
    text = re.sub(r'PorqueMeFui', 'Porque me fui', text)
    text = re.sub(r'NoMas', 'no mas', text)
    text = re.sub(r'nomas', 'no mas', text)
    text = re.sub(r'NuncaMás', 'Nunca Más', text)
    text = re.sub(r'NuncaMas', 'Nunca Más', text)
    text = re.sub(r'nuncamas', 'Nunca Más', text)
    text = re.sub(r'#', '', text)

    return text

def preprocessing_for_bert(data, tokenizer, max_len=300):
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
            text=text_preprocessing(sent),  # Preprocess sentence
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