import abc
import torch
import numpy as np
import pickle as pkl
from pytorch_pretrained_bert import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import transformers

class FinnishFeaturizer(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def featurize(self):
        pass

class BERTFeaturizer(FinnishFeaturizer):
    def __init__(self):
        self.model = transformers.BertForMaskedLM.from_pretrained("TurkuNLP/bert-base-finnish-uncased-v1") 
        '''
        self.model.eval()
        if torch.cuda.is_available(): #runs model on GPU
            self.model = self.model.cuda()
        '''
        self.tokenizer = transformers.BertTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-uncased-v1")

    def featurize(self, docs):
        self.model.eval() #was first done in Line 18/19 when initiating...
        dataset = BERTFeatureDataset(docs, self.tokenizer)
        print("Length of docs: ", len(docs))
        print("Length of dataset: ", len(dataset))

        feature_loader = DataLoader(
            dataset, shuffle=False, batch_size=30, collate_fn=self.__pad_collate)
            #dataset, shuffle=False, batch_size=30)

        doc_embeds = []
        # Predict hidden states features for each layer.
        #N.B. Masked LM output is different for non-base models (Section: MaskedLMOutput)
        print("Length of feature_loader: ", len(feature_loader)) #327 adp-cm / 30 is 11 
        for input_tok_ids, input_mask, input_seq_ids in feature_loader:
            #output = self.model(input_tok_ids, token_type_ids=input_seq_ids, attention_mask=input_mask)
            #print(type(output)) = tuple
            #print(len(output)) = 1

            #original code from pre-loaded mBERT (or BERT_base) Featurizer:
            '''
            all_encoder_layers = self.model(
                input_tok_ids, token_type_ids=input_seq_ids, attention_mask=input_mask)
            doc_embeds.append(all_encoder_layers[11][:, 0, :].detach())
            '''

            #new line from Prakash tutorial: https://towardsdatascience.com/3-types-of-contextualized-word-embeddings-from-bert-using-transfer-learning-81fcefe3fe6d

            '''
            #print(len(all_encoder_layers[1])) # = 13
            #first hidden state is input state, so remove first hidden state
            hidden_states = all_encoder_layers[1][1:]
            #Get embeddings from final BERT layer ([11] for the 12-hidden states)
            token_embeddings = hidden_states[-1]

            #Collapse tensor into 1-D
            token_embeddings = torch.squeeze(token_embeddings, dim=0)

            #Append each token to doc list
            doc_embeds.append(token_embeddings)
            '''

            all_encoder_layers = self.model(
                input_tok_ids, token_type_ids=input_seq_ids, attention_mask=input_mask,
                output_hidden_states=True)

            #first hidden state is input state, so remove first hidden state
            hidden_states = all_encoder_layers[1][1:]
            #print(len(hidden_states[11])) # = 30, last one is 27 (11 entries total in feature_loader)
            #token_embeds = torch.squeeze(hidden_states[11], dim=0)
            doc_embeds.append(hidden_states[11].detach())

        return torch.cat(doc_embeds)
    

    def __pad_collate(self, batch): 
        (token_ids, segment_ids, mask_ids) = zip(*batch) #reverses the order of masks and segments in feature_loader
        token_ids_pad = pad_sequence(token_ids, batch_first=True)
        mask_ids_pad = pad_sequence(mask_ids, batch_first=True)
        segment_ids_pad = pad_sequence(segment_ids, batch_first=True)
        return token_ids_pad, mask_ids_pad, segment_ids_pad

class BERTFeatureDataset(Dataset):
    '''
    Torch dataset class that pads sentences for vector processing by BERT
    '''

    def __init__(self, docs, tokenizer):
        self.docs = docs
        self.tokenizer = tokenizer
        self.all_tokens, self.all_token_ids, self.all_token_masks = self.tokenize_data(
            docs, self.tokenizer)
    
    def __len__(self) :
        return len(self.docs)

    def __getitem__(self, idx) :
        #return self.docs[idx]
        return self.all_tokens[idx], self.all_token_ids[idx], self.all_token_masks[idx]

    def tokenize_data(self, docs, tokenizer):

        all_tokens = []
        all_token_ids = []
        all_token_masks = []

        for doc in docs:
            doc = "[CLS] " + doc + " [SEP]"
            token_vector = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(doc))
            all_tokens.append(torch.tensor(token_vector)) #tokens_tensor
            all_token_ids.append(torch.tensor(
                [0] * len(token_vector)))   # BERT sentence ID 
            all_token_masks.append(torch.tensor(
                [1] * len(token_vector)))  # BERT Masking / segments tensor

        return all_tokens, all_token_ids, all_token_masks

class FinnishFeaturizerFactory:
    def featurize(self, doc_list, feature_type):
        featurizer = get_featurizer(feature_type)
        return featurizer.featurize(doc_list)
        
def get_featurizer(feature_type):
    if feature_type == 'BERT':
        return BERTFeaturizer()
    else:
        raise Exception("Invalid feature type")