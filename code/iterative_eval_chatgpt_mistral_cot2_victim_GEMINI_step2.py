import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import numpy as np
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from transformers import AutoTokenizer
# from datasets import load_dataset
# from sklearn.metrics import f1_score
import math


############3
import os
from google import genai
from google.genai import types

client_gemini = genai.Client(api_key="YOUR API KEY")

#############3

import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('-run', '--run', help='index of run', required=False)
# parser.add_argument('-seed', '--seed', type=int, help='seed', required=False)
# parser.add_argument('-order', '--order', type=int, help='order of in context examples', required=False)
# parser.add_argument('-target_type', '--target_type', type=str, help='noun_phrase, claim, or mixed', required=False)
# parser.add_argument('-model', '--model', type=str, default='gpt-4-1106-preview', help='noun_phrase, claim, or mixed', required=False)
parser.add_argument('-test_data', '--test_data', type=str, default='./', help='path of test data', required=True)
# parser.add_argument('-domain', '--domain', type=str, default='subtaskA', help='path of test data', required=False)
parser.add_argument('-tosave_path', '--tosave_path', type=str, default='./', help='path of test data', required=True)
parser.add_argument('-max_tries', '--max_tries', type=int, default=20, help='max num of retries for chatgpt', required=True)
parser.add_argument('-model', '--model', type=str, default='mistral', help='mistral, llama2, or chatgpt', required=True)
parser.add_argument('-max_token', '--max_token', type=int, default=300, help='', required=False)
parser.add_argument('-num_beams', '--num_beams', type=int, default=1, help='', required=False)
parser.add_argument('-ex_random_seed', '--ex_random_seed', type=int, default=1, help='', required=False)
parser.add_argument('-cot_step1_result', '--cot_step1_result', type=str, default='./annotation_results/rephrased_results_test_sacarsm_chatgpt_cot_testonly2_step1.csv', help='', required=False)
parser.add_argument('-model_in_use', '--model_in_use', type=str, default='BERTWEET_large', help='Bert,Bart_encoder,BERTWEET_large,Bert_large,XLNet_large', required=False)
parser.add_argument('-dataset_name', '--dataset_name', type=str, default='covid', help='covid, vast', required=False)



args = vars(parser.parse_args())
# device = torch.device("cpu")
device = 'cuda'

def detect_stance_2026(text, target):
    """
    Analyzes stance using the modern Gemini 2.0 Flash model.
    """
    
    # SYSTEM INSTRUCTION: Sets the "persona" and strict rules
    sys_instr = "You are a precise linguistic analyzer. You only output one word: 'favor', 'against', or 'neutral'."
#     sys_instr = """You are a stance detection classifier.

# Definition of stance toward a given target:

# FAVOR:
# The author explicitly or implicitly supports, agrees with, promotes, or positively evaluates the target.

# AGAINST:
# The author explicitly or implicitly opposes, criticizes, rejects, or negatively evaluates the target.

# NEUTRAL:
# The text is purely informational, descriptive, or does not show a clear positive or negative evaluation toward the target.



# Output exactly one word:
# favor
# against
# neutral"""

# Important rules:
# - Even mild approval counts as FAVOR.
# - Even mild criticism or concern counts as AGAINST.
# - Do NOT choose NEUTRAL if any evaluative language is present.
# - If the author shows any preference, it is not NEUTRAL.
    # USER PROMPT: The specific task
    user_prompt = f"Target: {target}\nText: {text}\n\nWhat is the author's stance towards the target?"

    try:
        response = client_gemini.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=sys_instr,
                temperature=0.0  # Set to 0 for maximum consistency
            ),
            contents=user_prompt
        )
        
        # Extract and clean result
        result = response.text.strip().lower().replace(".", "")
        print("user_prompt:",user_prompt)
        print("result:",result)
        return result

    except Exception as e:
        return f"Error: {e}"

class GoEmotionsDataset(torch.utils.data.Dataset):
    def __init__(self, text_list, text_list_raw, labels,targets, ids):
        self.text_list = text_list
        self.text_list_raw = text_list_raw
        print("self.text_list_raw[0]:",self.text_list_raw[0])
        self.labels = labels
        self.targets = targets
        self.tokenizer = None
        self.ids = ids

    def __getitem__(self, idx):

        # tok = self.tokenizer(
            # self.text_list[idx], padding='max_length', max_length=200, truncation=True)
        tok = self.tokenizer(
            [' '.join(self.text_list[idx]), ' '.join(self.targets[idx])], padding='max_length', max_length=200, truncation=True)
        item = {key: torch.tensor(tok[key]) for key in tok}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item

    def __len__(self):
        return len(self.labels)
        return len(self.labels_list)


# def evaluate(model, test_dataloader, criterion):
#     full_predictions = []
#     true_labels = []

#     model.eval()
#     crt_loss = 0

#     for elem in test_dataloader:
#         x = {key: elem[key].to(device)
#              for key in elem if key not in ['text', 'idx']}
#         logits = model(input_ids=x['input_ids'], token_type_ids=x['token_type_ids'], attention_mask=x['attention_mask'])
#         results = torch.argmax(logits.logits, dim=1)

#         crt_loss += criterion(logits.logits, x['labels']
#                               ).cpu().detach().numpy() / 8
#         full_predictions = full_predictions + \
#             list(results.cpu().detach().numpy())
#         true_labels = true_labels + list(elem['labels'].cpu().detach().numpy())


#     true_labels = np.argmax(true_labels, axis=1)
#     model.train()

#     return f1_score(true_labels, full_predictions, average='weighted'), f1_score(true_labels, full_predictions, average='macro'), (crt_loss / len(test_dataloader)) / 8


def single_label_no_neutral(dataset_goemotions, num_classes=3):
    datasets = []
    text,text_raw,labels,targets = dataset_goemotions[0],dataset_goemotions[1],dataset_goemotions[2],dataset_goemotions[3]


    assert len(text) == len(labels)

    new_text, new_text_raw, new_labels, new_ids, new_targets = [], [], [], [], []
    for i in range(len(text)):
        new_text.append(text[i])
        new_text_raw.append(text_raw[i])
        new_labels.append(labels[i])
        new_ids.append(i)
        new_targets.append(targets[i])

    assert len(new_text) == len(new_labels)

    datasets = GoEmotionsDataset(new_text, new_text_raw, new_labels, new_targets, new_ids)

    return datasets


def preprocess_simple(ds):
    for i in range(len(ds.labels)):
        lbl = ds.labels[i]
        ds.labels[i] = np.zeros(3)
        ds.labels[i][lbl] = 1
    return ds


##################################################################################
##################################################################################
import preprocessor as p
import re
import wordninja
import csv
import pandas as pd
# from utils import augment

def load_data(filename,mode,trainvaltest):

    concat_text = pd.DataFrame()

    if mode=='train_en_test_en':
        encoding='ISO-8859-1'
        label_words=['AGAINST','FAVOR','NONE']
        # label_words=['AGAINST','FAVOR','NONE']

    df = pd.read_csv(filename)
    print(filename,"df.columns:",df.columns,'Text' in df.columns)
    print("df:",df)
    try:
        raw_text = pd.read_csv(filename,usecols=['Tweet'], encoding=encoding)
    except:
        raw_text = pd.read_csv(filename,usecols=['Text'], encoding=encoding)

    raw_target = pd.read_csv(filename,usecols=['Target 1'], encoding=encoding)
    raw_label = pd.read_csv(filename,usecols=['Stance 1'], encoding=encoding)
    label = pd.DataFrame.replace(raw_label,label_words, [0,1,2])

    if 'vast' in filename:
        seen = pd.read_csv(filename,usecols=['seen?'], encoding=encoding)
        concat_text = pd.concat([raw_text, label, raw_target, seen], axis=1)
    else:
        concat_text = pd.concat([raw_text, label, raw_target], axis=1)       


    concat_text.rename(columns={'Stance 1':'Stance','Target 1':'Target'}, inplace=True)

    if 'vast' in filename and 'train' not in filename:
        concat_text = concat_text[concat_text['seen?'] != 1] # remove few-shot labels

    print(filename,"concat_text.columns:",concat_text.columns)
    return concat_text


# Data Cleaning
def data_clean(strings, norm_dict):

    p.set_options(p.OPT.URL,p.OPT.EMOJI,p.OPT.RESERVED)
    clean_data = p.clean(strings) # using lib to clean URL,hashtags...
    clean_data = re.sub(r"#SemST", "", clean_data)
    clean_data = re.findall(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+",clean_data)
    clean_data = [[x.lower()] for x in clean_data]

    for i in range(len(clean_data)):
        if clean_data[i][0] in norm_dict.keys():
            clean_data[i] = norm_dict[clean_data[i][0]].split()
            continue
        if clean_data[i][0].startswith("#") or clean_data[i][0].startswith("@"):
            clean_data[i] = wordninja.split(clean_data[i][0]) # separate hashtags
    clean_data = [j for i in clean_data for j in i]

    return clean_data


# Clean All Data
def clean_all(filename, norm_dict, mode,trainvaltest,prompt_index):
  concat_text = load_data(filename, mode,trainvaltest) # load all data as DataFrame type

  try:
    raw_data = concat_text['Tweet'].values.tolist() # convert DataFrame to list ['string','string',...]
  except:
    raw_data = concat_text['Text'].values.tolist() # convert DataFrame to list ['string','string',...]

  label = concat_text['Stance'].values.tolist()
  x_target = concat_text['Target'].values.tolist()
  clean_data = [None for _ in range(len(raw_data))]

  for i in range(len(raw_data)):
    if prompt_index==0:
      x_target[i] = x_target[i]



    clean_data[i] = data_clean(raw_data[i],norm_dict) # clean each tweet text [['word1','word2'],[...],...]
    x_target[i] = data_clean(x_target[i],norm_dict)
    # print("raw_data:",raw_data[0],len(raw_data))
    # print(bk)
    # clean_data[i] = raw_data[i].split(' ')
    # x_target[i] = x_target[i]


  avg_ls = sum([len(x) for x in clean_data])/len(clean_data)


  print(100*"#")
  print("average length: ", avg_ls)
  print("num of subset: ", len(label))
  print("x_target[0]:",x_target[0])
  print(100*"#")
  return clean_data,raw_data,label,x_target

##################################################################################
##################################################################################
import torch
import torch.nn as nn
from transformers import BertModel, BartConfig, BartForSequenceClassification
from transformers.models.bart.modeling_bart import BartEncoder, BartPretrainedModel
from transformers import RobertaModel, MT5EncoderModel, MT5Model,MT5ForConditionalGeneration,AutoModel

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification,MBartForSequenceClassification
from transformers import XLNetModel,XLNetForSequenceClassification
class bertweet_large_classifier(nn.Module):

    def __init__(self, num_labels,  dropout=0.1):

        super(bertweet_large_classifier, self).__init__()

        self.relu = nn.ReLU()

        self.roberta = model = AutoModel.from_pretrained('vinai/bertweet-base')
        self.roberta.pooler = None
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # self.linear = nn.Linear(1024*2, 1024)
        self.linear = nn.Linear(self.roberta.config.hidden_size, self.roberta.config.hidden_size)

        self.eos_token_id=2

        print(200*"!")
        print("Working with a {}-class classification task".format(num_labels))
        print(200*"!")

        self.out = nn.Linear(self.roberta.config.hidden_size, num_labels)         

    def forward(self, **kwargs):
        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']        
        # print("x_input_ids:",x_input_ids.size())
        # print("x_atten_masks:",x_atten_masks.size())

        last_hidden = self.roberta(input_ids=x_input_ids, attention_mask=x_atten_masks)

        # print("last_hidden:",last_hidden)
        # 
        cls_hidden = last_hidden[0][:, 0, :]
        # print(bk)

        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        return out
# BERT
class bert_large_classifier(nn.Module):

    def __init__(self, num_labels, dropout=0.1):

        super(bert_large_classifier, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.bert = BertModel.from_pretrained("bert-large-uncased")
        self.bert.pooler = None
        # self.linear = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, **kwargs):

        x_input_ids, x_atten_masks, x_seg_ids = kwargs['input_ids'], kwargs['attention_mask'], kwargs['token_type_ids']
        last_hidden = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)
        ###########################################################################################
        #                                       用CLS token
        ###########################################################################################
        cls_hidden = last_hidden[0][:, 0, :]
        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        return out
# xlnet
class xlnet_large_classifier(nn.Module):

    def __init__(self, num_labels, dropout=0.1):

        super(xlnet_large_classifier, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.bert = XLNetModel.from_pretrained('xlnet-large-cased')
        # self.bert = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased')

        self.bert.pooler = None
        # self.linear = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, **kwargs):

        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
        output = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks)

        ###########################################################################################
        #                                       用CLS token
        ###########################################################################################
        cls_hidden = output.last_hidden_state[:, 0, :]
        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        return out
  # BERT
class bert_classifier(nn.Module):
    def __init__(self, num_labels):

        super(bert_classifier, self).__init__()

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.pooler = None
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, **kwargs):

        x_input_ids, x_atten_masks, x_seg_ids = kwargs['input_ids'], kwargs['attention_mask'], kwargs['token_type_ids']
        last_hidden = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)

        cls_hidden = last_hidden[0][:, 0, :]
        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)



        return out
# BART
# class Encoder(BartPretrainedModel):

#     def __init__(self, config: BartConfig):

#         super().__init__(config)

#         padding_idx, vocab_size = config.pad_token_id, config.vocab_size
#         self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
#         # self.encoder = BartEncoder(config, self.shared)
#         self.encoder = BartEncoder(config)
class Encoder(BartPretrainedModel):

    # _tied_weights_keys = ["encoder.embed_tokens.weight"]
    _tied_weights_keys = {
        "encoder.embed_tokens.weight": "shared.weight"
    }
    def __init__(self, config: BartConfig):

        super().__init__(config)

        padding_idx = config.pad_token_id
        vocab_size = config.vocab_size

        # Shared embedding
        self.shared = nn.Embedding(
            vocab_size,
            config.d_model,
            padding_idx
        )

        # Encoder
        self.encoder = BartEncoder(config)

        # Tie embeddings manually
        self.encoder.embed_tokens = self.shared

        # REQUIRED by new HF versions
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = value

    def forward(self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs

class bart_mnli_encoder_classifier(nn.Module):

    def __init__(self, num_labels, dropout=0.1):

        super(bart_mnli_encoder_classifier, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.config = BartConfig.from_pretrained('facebook/bart-large-mnli')
        self.bart = Encoder.from_pretrained("facebook/bart-large-mnli")
        self.bart.pooler = None
        # self.linear = nn.Linear(self.bart.config.hidden_size*2, self.bart.config.hidden_size)
        self.linear = nn.Linear(self.bart.config.hidden_size, self.bart.config.hidden_size)
        self.out = nn.Linear(self.bart.config.hidden_size, num_labels)

    def forward(self, **kwargs):

        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
        last_hidden = self.bart(input_ids=x_input_ids, attention_mask=x_atten_masks)

        cls_hidden = last_hidden[0][:, 0, :]
        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        return out
##################################################################################
##################################################################################
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification,BertTokenizer
from transformers import BertTokenizer, AutoTokenizer, BertweetTokenizer, BartTokenizer,RobertaTokenizer,XLNetTokenizer
from transformers import RobertaModel, MT5EncoderModel, MT5Model,MT5ForConditionalGeneration,AutoModel
import json

# model_in_use = 'BERTWEET'
# model_in_use = 'RoBERTa_base'
# model_in_use = 'XLNet_base'

# model_in_use = 'Bart_encoder'
# model_in_use = 'BERTWEET_large'
# model_in_use = 'Bert_large'
# model_in_use = 'XLNet_large'
model_in_use = args['model_in_use']
dummy_dataset_name = "pstance"

if model_in_use=='Bert':
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    config_file = "./configs/config-bert_base.txt"
# elif model_in_use == 'BERTWEET':
#     tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
#     config_file = "config-bertweet_base.txt"
# elif model_in_use=='Bart_encoder':
#     tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-mnli", normalization=True)
#     config_file = "config-bart_large_mnli_encoder.txt"
# elif model_in_use == 'XLNet_base':
#     tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
#     config_file = "config-xlnet_base.txt"
# elif model_in_use=='RoBERTa_base':
#     tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#     config_file = "config-roberta_base.txt"
elif model_in_use=='Bart_encoder':
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-mnli", normalization=True)
    config_file = "./configs/config-bart_large_mnli_encoder_tune1.txt"
elif model_in_use == 'BERTWEET_large':
    # tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large", normalization=True)
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
    config_file = "./configs/config-bertweet_base_tune1.txt"
elif model_in_use == 'Bert_large':
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True)
    config_file = "./configs/config-bert_large_tune1.txt"
elif model_in_use == 'XLNet_large':
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    config_file = "./configs/config-xlnet_large_tune1.txt"
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
# model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

import os
base_path='./'

with open(os.path.join(base_path,"noslang_data.json"), "r") as f:
    data1 = json.load(f)
data2 = {}
with open(os.path.join(base_path,"emnlp_dict.txt"),"r") as f:
    lines = f.readlines()
    for line in lines:
        row = line.split('\t')
        data2[row[0]] = row[1].rstrip()
norm_dict = {**data1,**data2}

### load config file
with open(os.path.join(config_file), 'r') as f:
    config = dict()
    for l in f.readlines():
        config[l.strip().split(":")[0]] = l.strip().split(":")[1]
# model_select = config['model_select']
config_max_length = int(int(config['max_tok_len']) + int(config['max_tar_len']))
print("config:",config)



### data loader
# test_data_file = os.path.join(base_path,"datasets","covid19","raw_test_all_onecol_no_neutral.csv")
# test_data_file = os.path.join(base_path,"datasets","pstance","raw_test_all_onecol.csv")

# test_data_file = os.path.join(base_path,"datasets","covid19","raw_test_all_onecol.csv")
test_data_file = args['test_data']
x_test, x_test_raw, y_test, x_test_target = clean_all(test_data_file, norm_dict,'train_en_test_en','test',0)
x_test_all = [x_test,x_test_raw, y_test,x_test_target]


ds_test = single_label_no_neutral(x_test_all)
ds_test = preprocess_simple(ds_test)
ds_test.tokenizer = tokenizer
##################################################################################
# load the saved model
from transformers import AutoModelForSequenceClassification,BertTokenizer
from transformers import BertTokenizer, AutoTokenizer, BertweetTokenizer, BartTokenizer,RobertaTokenizer,XLNetTokenizer
from transformers import RobertaModel, MT5EncoderModel, MT5Model,MT5ForConditionalGeneration,AutoModel

####################################################################################################
# path_checkpoint = '/data/czhao43/adv_stance/EMNLP2022_BART_MNLI_w_tensorboard_w_earlystopping/src/'
path_checkpoint = './'
# dataset_name = 'covid'
dataset_name = args['dataset_name']

### Bert for covid
if model_in_use=='Bert':
  checkpoint_path = os.path.join(path_checkpoint,"checkpoints",dataset_name,model_in_use,"checkpoint0.pt")
  model = bert_classifier(2)
# elif model_in_use=='RoBERTa_base':
#   checkpoint_path = os.path.join(path_checkpoint,"checkpoints","covid_roberta","checkpoint0.pt")
#   model = roberta_base_classifier(3)
# elif model_in_use=='XLNet_base':
#   checkpoint_path = os.path.join(path_checkpoint,"checkpoints","covid_xlnet","checkpoint3.pt")
#   model = xlnet_base_classifier(3)
# elif model_in_use=='BERTWEET':
#   checkpoint_path = os.path.join(path_checkpoint,"checkpoints","covid_bertweet","checkpoint3.pt")
#   model = bertweet_base_classifier(3)



elif model_in_use=='Bart_encoder':
  # checkpoint_path = os.path.join(path_checkpoint,"checkpoints",dataset_name,model_in_use+"1","checkpoint0.pt")
  checkpoint_path = os.path.join(path_checkpoint,"checkpoints",dummy_dataset_name,model_in_use,"checkpoint0.pt")
  model = bart_mnli_encoder_classifier(2)
elif model_in_use=='BERTWEET_large':
  # checkpoint_path = os.path.join(path_checkpoint,"checkpoints",dataset_name,model_in_use+"1","checkpoint0.pt")
  checkpoint_path = os.path.join(path_checkpoint,"checkpoints",dataset_name,"BERTWEET","checkpoint0.pt")
  model = bertweet_large_classifier(2)
elif model_in_use=='Bert_large':
  checkpoint_path = os.path.join(path_checkpoint,"checkpoints",dataset_name,model_in_use,"checkpoint0.pt")
  model = bert_large_classifier(3)
elif model_in_use=='XLNet_large':
  checkpoint_path = os.path.join(path_checkpoint,"checkpoints",dataset_name,model_in_use,"checkpoint0.pt")
  model = xlnet_large_classifier(3)
# elif model_in_use=='Bart_encoder':
#   checkpoint_path = os.path.join(base_path,"checkpoints","covid_bart","checkpoint0.pt")
#   model = roberta_classifier(3)
####################################################################################################


print("checkpoint_path:",checkpoint_path)

# checkp = torch.load(checkpoint_path, map_location='cpu')
checkp = torch.load(checkpoint_path)
state_dict = checkp.state_dict() if hasattr(checkp, 'state_dict') else checkp
new_checkp = {key.replace("module.", ""): value for key, value in checkp.items()}

print(150*"#")
# Print the keys
for key in new_checkp.keys():
    print(key)
# print(bk)
print(150*"#")
if "roberta.embeddings.position_ids" in new_checkp:
    del new_checkp["roberta.embeddings.position_ids"]
model.load_state_dict(new_checkp)
print(100*"#")
print("model loaded from checkpoint: {}".format(checkpoint_path))
print(100*"#")

##################################################################################
# stance classes
emotions = [
"against",
"favor",
"neutral"
]
##################################################################################
##################################################################################
import torch.nn as nn
from scipy.special import softmax

def single_example_predictions(model, tokenizer, text,target):
  # tok = tokenizer(' '.join(text), ' '.join(target), padding='max_length', max_length=config_max_length, truncation=True)
  # tok = {k : torch.tensor(tok[k]).unsqueeze(dim=0) for k in tok}
  # print("tok:",tok.keys())
  # # print("tok['input_ids'].size():",tok['input_ids'].size())
  # # print(bk)
  # model.eval()
  # with torch.no_grad():
  #   try:
  #       prediction = model(input_ids=tok['input_ids'], token_type_ids=tok['token_type_ids'], attention_mask=tok['attention_mask']).squeeze(0).numpy()        
  #   except:
  #       prediction = model(input_ids=tok['input_ids'], attention_mask=tok['attention_mask']).squeeze(0).numpy()

  #   predicted_class = np.argmax(prediction)

  # model.train()

  text = " ".join(text)
  target = " ".join(target)

  print("text:",text)
  print("target:",target)
  stance = detect_stance_2026(text, target)

  print("stance:",stance)

  dummy_softmax_prediction = [0.9554061,  0.04459386,  0.04459386]
  dummy_predicted_class = 0
  # stance = 'favor'
  return stance, dummy_softmax_prediction, dummy_predicted_class
##################################################################################
##################################################################################
single_example_predictions(model, tokenizer, ["I", "hate", "it"],["she","love","them"])
##################################################################################
##################################################################################
# Annotator's name
name = 'Chenye'
# DO NOT CHANGE
start_id = 0
num_examples = 50
##################################################################################
##################################################################################
print("full test set:",len(ds_test))
# ids = set(ds_test.ids[start_id:(start_id + num_examples)])
ids = set(ds_test.ids)

to_annotate_examples = []

for i, id in enumerate(ds_test.ids):
  if id in ids:
    to_annotate_examples.append({'text' : ds_test.text_list[i],'text_raw' : ds_test.text_list_raw[i], 'label' : ds_test.labels[i], 'id' : ds_test.ids[i], 'target' : ds_test.targets[i]})
##################################################################################
##################################################################################
import json
import os
import time
from openai import OpenAI

# client = OpenAI(api_key="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")
client = OpenAI(api_key="YOUR API KEY")
import random
print('Retrieving Annotation Status')
path = './annotation_results/pstance_annotation' + str(start_id) + '_' + str(start_id + num_examples) + '_' + name + '.json'
if not os.path.exists(path):
  print('No annotations found for current task')
  d = {}
else:
  with open(path) as f:
    d = json.load(f)
  print('Loaded', len(d), 'annotations.')
##########################################################################################
def ask_chatgpt(text, target, stance, text_last,df_examples,args,analysis):
    def ChatEngine(prompt,instruction):
        # openai.api_key="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" #zcy8863
         #czhao43
        # openai.api_key="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" #isziwei
        response=client.chat.completions.create(model="gpt-3.5-turbo-0125",
        # model="gpt-4-0125-preview",
        messages=[
            # {"role": "system", "content":instruction},
            {"role": "user", "content": prompt},
        ])
        return response.choices[0].message.content 


    max_retries=5

    ###########################################################
    #   get in-context examples
    ###########################################################
    # print("df_examples1:",df_examples)
    # pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.max_colwidth', None)

    # print("df_examples2:",df_examples)
    # print(bk)
    random_row = df_examples.sample(n=1, random_state=args['ex_random_seed'])
    ex_text = random_row['Text'].values[0]
    ex_target = random_row['Target 1'].values[0]
    ex_stance = random_row['Stance 1'].values[0]
    ex_rephrased_text = random_row['Rephrased text'].values[0]
    # print("random_row:",random_row)
    # print("ex_text:",ex_text)
    # print(bk)
    ###########################################################
    if text_last!="":
        prompt="""Rephrase the following text to subtly express the same stance towards the specified target. The rephrasing should aim to convey the message in a manner that is not straightforward for stance detection algorithms to classify. Consider using sarcasm, figurative language, synonyms, similar expressions, indirect language, metaphors, or summarizing the sentiment in a nuanced way that retains the original stance towards the target but changes the sentence structure and vocabulary significantly. Please ensure the rephrased text remains coherent and can be understood by a human reader as maintaining the original stance towards the given text without directly mimicking the original text's style or terminology.
        Before rephrasing, a detailed analysis based on the semantic of the original text and the stance correlation between the text and the target will be provided. This analysis should guide the rephrasing process, ensuring that the nuanced stance and semantic meaning is effectively communicated through alternative expressions and structures.

        Original Text: {}
        Target: {}
        Stance: {}
        Semantic Analysis and Stance Correlation: {}
        Remember, the rephrased version should be creative and indirect enough to potentially obscure the stance from automated classifiers while remaining clear to a human audience.
        Here is a bad example, where the rephrased text is too similar to the original text: "{}". """.format(text, target, stance, analysis, text_last)
    else:
        prompt="""Rephrase the following text to subtly express the same stance towards the specified target. The rephrasing should aim to convey the message in a manner that is not straightforward for stance detection algorithms to classify. Consider using sarcasm, figurative language, synonyms, similar expressions, indirect language, metaphors, or summarizing the sentiment in a nuanced way that retains the original stance towards the target but changes the sentence structure and vocabulary significantly. Please ensure the rephrased text remains coherent and can be understood by a human reader as maintaining the original stance towards the given text without directly mimicking the original text's style or terminology.
        Before rephrasing, a detailed analysis based on the semantic of the original text and the stance correlation between the text and the target will be provided. This analysis should guide the rephrasing process, ensuring that the nuanced stance and semantic meaning is effectively communicated through alternative expressions and structures.

        Original Text: {}
        Target: {}
        Stance: {}
        Semantic Analysis and Stance Correlation: {}
        Remember, the rephrased version should be creative and indirect enough to potentially obscure the stance from automated classifiers while remaining clear to a human audience.""".format(text, target, stance, analysis)

    print(100*"-")
    print("prompt:",prompt)
    # print(bk)
    print(100*"-")
    instruction = ""
    for attempt in range(1, max_retries + 1):
        try:
            rephrased_text = ChatEngine(prompt,instruction)
            # print(f"Attempt {attempt}: Success")
            time.sleep(random.randint(1,3))
            return rephrased_text
        except Exception as e:
            print(f"Attempt {attempt}: API call failed with error '{e}'. Retrying...")
            time.sleep(random.randint(1,3))  # 介于重试之间的短暂暂停，避免过快重试
            if attempt == max_retries:
                print("Maximum retries reached. Giving up.")
                return 'GPT_FAILURE'


##########################################################################################

print("to_annotate_examples:",to_annotate_examples[0],len(to_annotate_examples))
# print(bk)
if args['model']=='llama2':
  ####################################################################
  #     load tokenizer and model for llama2
  ####################################################################
  MY_TOKEN="YOUR TOKEN"

  # pretrained_model_name_llama = "meta-llama/Llama-2-13b-chat-hf"
  pretrained_model_name_llama = "meta-llama/Llama-2-13b-chat-hf"
  model_llm = AutoModelForCausalLM.from_pretrained(pretrained_model_name_llama, device_map="auto", torch_dtype="auto",token=MY_TOKEN)
  tokenizer_llm = AutoTokenizer.from_pretrained(pretrained_model_name_llama,token=MY_TOKEN)
elif args['model']=='mistral':
  ####################################################################
  #     load tokenizer and model for mistral
  ####################################################################
  pretrained_model_name_mistral = "mistralai/Mistral-7B-Instruct-v0.2"
  model_llm = AutoModelForCausalLM.from_pretrained(pretrained_model_name_mistral, device_map="auto", torch_dtype="auto")
  tokenizer_llm = AutoTokenizer.from_pretrained(pretrained_model_name_mistral)
  tokenizer_llm.padding_side = "left"
  ####################################################################

####################################################################
#     load in-context examples
#################################################################### 
df_in_context_examples = pd.read_csv('./in_context_examples_human_selection.csv',encoding='utf-8')
print("df_in_context_examples:",df_in_context_examples,len(df_in_context_examples),df_in_context_examples.columns)
####################################################################
#     load step1 analysis
#################################################################### 
df_cot_analysis = pd.read_csv(args['cot_step1_result'])
print("df_cot_analysis:",df_cot_analysis,len(df_cot_analysis),df_cot_analysis.columns)
index_cot_step1 = 0
pd.set_option('display.max_colwidth', None)
####################################################################
idx = 0
#################################################################
#       debug only
#################################################################
# to_annotate_examples = to_annotate_examples[:20]
# args['max_tries'] = 2
#################################################################
for idx, elem in enumerate(to_annotate_examples):
  # idx+=1
  # if idx>100:
  #   continue
  # if idx<300:
  #   continue  
  # if idx>10:
  #   continue
  print(10*"-")
  print("elem['label']:",elem['label'])
  print("emotions[np.argmax(elem['label'])]:",emotions[np.argmax(elem['label'])])
  print(10*"-")
  if emotions[np.argmax(elem['label'])]=='neutral':
    continue
  # if str(elem['id']) not in d:
  d[elem['id']] = {}

  prediction, probabilities, predicted_class = single_example_predictions(model, tokenizer, elem['text'], elem['target'])
  print(10*"-")
  print("prediction:",prediction)
  print("emotions[np.argmax(elem['label'])]:",emotions[np.argmax(elem['label'])])
  print(10*"-")
  if prediction != emotions[np.argmax(elem['label'])]:
    d[elem['id']] = None
    continue

  gold_label_idx = np.argmax(elem['label'])

  print(50*'-')
  # print('Text:', ' '.join(elem['text']))
  print('Raw Text:', elem['text_raw'])
  print('Target:', ' '.join(elem['target']))
  print('Stance:', prediction)
  print('gold_label_idx:', gold_label_idx)

  print('Model probability:', probabilities[gold_label_idx])
  print(50*'-')
  print('Rephrase the text accordingly')

  rephrased0 = [' '.join(elem['text'])]

  max_try = args['max_tries']
  cnt_tries = 0
  results = []

  # analysis = df_cot_analysis.loc[[index_cot_step1]]['Rephrased Text'].values[0]
  # text_raw_check = df_cot_analysis.loc[[index_cot_step1]]['Text'].values[0]
  # index_cot_step1+=1

  analysis = df_cot_analysis.loc[[idx]]['Rephrased Text'].values[0]
  text_raw_check = df_cot_analysis.loc[[idx]]['Text'].values[0]
  print(100*"#")
  print("text_raw_check:",text_raw_check,idx)
  print(100*"#")
  print("elem['text_raw']:",elem['text_raw'],idx)
  print(100*"#")
  assert text_raw_check==elem['text_raw']

  # print("analysis:",analysis)
  # print(bk)
  while True:
    rephrased = [elem['text_raw'],' '.join(elem['target']),prediction,probabilities[gold_label_idx]]
    if cnt_tries==0:
      text_last=""
    else:
      text_last=inp
    if args['model']=='chatgpt':
      inp = ask_chatgpt(elem['text_raw'],' '.join(elem['target']),prediction,text_last,df_in_context_examples,args,analysis)
    elif args['model']=='llama2':
      inp = ask_llama2(model_llm,tokenizer_llm,elem['text_raw'],' '.join(elem['target']),prediction,text_last)
    elif args['model']=='mistral':
      inp = ask_mistral(model_llm,tokenizer_llm,elem['text_raw'],' '.join(elem['target']),prediction,text_last)

    print(100*"#")
    print("elem['text_raw']:",elem['text_raw'])
    print(100*"=")
    print("inp:",inp)
    print(100*"=")
    # print(bk)
    if inp.lower() == 'next':
      # rephrased.append('NEXT')
      break
    else:
      rephrased0.append(inp)
      cleaned_inp = data_clean(inp,norm_dict)
      new_prediction, new_probabilities, new_predicted_class= single_example_predictions(model, tokenizer, cleaned_inp,elem['target'])

      rephrased.append(inp)
      rephrased.append(new_prediction)
      rephrased.append(new_probabilities[np.argmax(new_probabilities)])
      rephrased.append(new_probabilities[gold_label_idx])

      if new_prediction != prediction:
        print(10*"*",'  Success!   ',10*"*", cnt_tries)
        print(new_prediction + ':', new_probabilities[np.argmax(new_probabilities)], prediction + ':', new_probabilities[gold_label_idx])
        rephrased.append('Success')
        results.append(rephrased)
        break
      else:
        print(10*"*",'  The tweet needs further rephrasing, keep trying!   ',10*"*", cnt_tries)
        print(prediction + ':', new_probabilities[gold_label_idx])
        rephrased.append('Failure')
        results.append(rephrased)
    cnt_tries+=1
    if cnt_tries>max_try:
      break
  # print("rephrased:",rephrased)
  d[elem['id']][prediction] = rephrased0
  with open(path, 'w') as f:
    json.dump(d, f)

  file_path = os.path.join(args['tosave_path'])
  header = not os.path.exists(file_path)
  results_df = pd.DataFrame(results,columns=['Text','Target 1', 'Stance 1', 'Prob', 'Rephrased Text', 'Stance 2', 'Prob on Stance 2', 'Prob on Stance 1','Success?'])
  results_df.to_csv(file_path,index=False,header=header,mode='a')
  print(file_path, 'save, done!')
print('Done!')
