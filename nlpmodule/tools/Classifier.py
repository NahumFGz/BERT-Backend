import pandas as pd

from textwrap import wrap

from nlpmodule.tools.LoadModel import bert_classifier
from nlpmodule.tools.Preprocessing import preprocessing_for_bert
from nlpmodule.tools.Predict import bert_predict

def classifier_treatment_load(text, max_len=300, batch_size=16):
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
    
    probs = bert_predict(bert_classifier(), test_dataloader)
    preds = probs[:, 1]

    # Get accuracy over the test set
    id_y_pred = probs.argmax(axis=1)
    print("\n".join(wrap(text)))

    #print(f"Expert: {label}")
    #print(f"Model : {labels[id_y_pred[0]]} - {probs[0][id_y_pred[0]]:.3f}")
    return {
        "text": text,
        "label_predict": labels[id_y_pred[0]],
        "label_probability": float(probs[0][id_y_pred[0]])
    }