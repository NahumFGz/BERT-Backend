import torch

#def bert_classifier(path='./nlpmodule/weights/Twitter_model1.pth'):
#    return torch.load(path, map_location='cpu')

def bert_classifier(path='./nlpmodule/weights/NewPreprocesing_MDLBETO_BLCnormal_TS0.2_RS2020_epch10_lr5e-05_model1.pth'):
    return torch.load(path, map_location='cpu')