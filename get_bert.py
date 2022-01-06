import pandas as pd
from Featurizer import FeaturizerFactory
from LatinFeaturizer import LatinFeaturizerFactory
from FinnishFeaturizer import FinnishFeaturizerFactory
import os
import torch
'''
Column Names for Full Annotation Files:

sentence_no	
word	
word_transl	
adp_case	
scene_role	
fn_role

Column Names for Adpositions and Case Markers Only Files:

adp_cm = adpositions and case markers

'''

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_bert(filetag, data, lang) :
    print("Extracting BERT embeddings")
    adp_cm_tokens = data["adp_cm"] #Access Clause column from master data DataFrame

    #Pick BERT model
    if lang == "Finnish" :
        featurizer = FinnishFeaturizerFactory() #uncased FinBERT
    elif lang == "Latin" :
        featurizer = LatinFeaturizerFactory()
    else : #language = both (multilingual)
        featurizer = FeaturizerFactory()

    print("Featurizing")
    X = featurizer.featurize(adp_cm_tokens, "BERT") #X is a PyTorch tensor

    '''Error for filtered_X line...

    IndexError: The shape of the mask [327, 768] at index 1 
    does not match the shape of the indexed tensor [327, 6, 768] at index 1'''
    #filtered_X = X[~torch.any(X.isnan(),dim=1)]

    #torch.save(filtered_X, filetag + "_" + "x_tensor.pt")
    torch.save(X, filetag + "_" + "x_tensor.pt")

    #Uncomment to load pre-saved BERT embeddings
    #filtered_X = torch.load("x_tensor.pt")
    #return filtered_X
    return X #Pytorch tensor


if __name__ == "__main__" :
    #Read in Pandas DataFrame objects for each file 
    #that contains only adpositions and case marker morphemes
    pp = pd.read_csv("finbert_adpositions_only/pp_final_labels.csv")
    r = pd.read_csv("mbert_adpositions_only/r_final_labels.csv")

    ui = input("Finnish, Latin, or both (mBERT): ")
    if ui == "Finnish" or ui == "both" :
        pp_x = get_bert("pp", pp, ui)
    elif ui == "Latin" or ui == "both" :
        r_x = get_bert("r", r, ui)