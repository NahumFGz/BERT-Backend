from nlpmodule.tools.BertClassifier import BertClassifier
from nlpmodule.tools.Classifier import classifier_treatment_load

def main():
    text   = 'She is gentle and kind, she never disrespected me'
    result = classifier_treatment_load(text)
    print(result)

if __name__ == '__main__':
    main()