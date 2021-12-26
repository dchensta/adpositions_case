import pandas as pd
from Featurizer import FeaturizerFactory
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

def get_bert(filetag, data) :
    print("Extracting BERT embeddings")
    adp_cm_tokens = data["adp_cm"] #Access Clause column from master data DataFrame
    featurizer = FeaturizerFactory()
    print("Featurizing")
    X = featurizer.featurize(adp_cm_tokens, "BERT")
    filtered_X = X[~torch.any(X.isnan(),dim=1)]
    torch.save(filtered_X, filetag + "_" + "x_tensor.pt")

    #Uncomment to load pre-saved BERT embeddings
    #filtered_X = torch.load("x_tensor.pt")
    return filtered_X

if __name__ == "__main__" :
    #Read in Pandas DataFrame objects for each file 
    #that contains only adpositions and case marker morphemes
    pp = pd.read_csv("adpositions_only/pp_final_labels.csv")
    r = pd.read_csv("adpositions_only/r_final_labels.csv")

    pp_x = get_bert("pp", pp)
    r_x = get_bert("r", r)