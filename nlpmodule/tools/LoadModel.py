import torch

def bert_classifier(path='./nlpmodule/weights/Twitter_model1.pth'):
    return torch.load(path, map_location='cpu')