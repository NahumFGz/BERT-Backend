import torch

def bert_classifier(path='./weights/model1.pth'):
    return torch.load(path, map_location='cpu')