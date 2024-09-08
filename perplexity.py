from collections import defaultdict
import numpy as np
import pandas as pd
import random
import re
from tqdm import tqdm
import torch

np.random.seed(42)
random.seed(42)

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1.5", 
                                             torch_dtype="auto", 
                                             trust_remote_code=True,
                                             output_hidden_states=True,
                                             cache_dir='./cache/'
                                            )
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1.5", trust_remote_code=True, cache_dir='./cache/')

# model = LlamaForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",
#                                              torch_dtype=torch.float16,
#                                              cache_dir='./cache/'
#                                             )
# tokenizer = LlamaTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", cache_dir='./cache/')

device = "cuda"
model = model.to(device)

def clean_string(input_string):
    input_string_2 = re.sub(r"\n" , " ", input_string)
    input_string_2 = re.sub(r"[^A-Za-z0-9 !\"$%&\'()\*+,-./:;?@^_`~]" , "", input_string_2)
    input_string_2 = re.sub(r"[ ]+", " ", input_string_2)
    input_string_2 = input_string_2.strip()
    input_string_2
    return input_string_2


df = pd.read_csv("chatgpt_filtered.csv")
X = [clean_string(df["prompt_body"][i]) + "_SEP_" + clean_string(df["gen_body"][i])
     for i in range(len(df))]
y = [df["label"][i] for i in range(len(df))]

def get_nlls(sentences):
    
    nlls_full = []
    nlls_sentences = []
    prev_encodings = None
    
    for sentence in sentences:
        encodings = tokenizer(sentence, return_tensors="pt")
        max_length = 2
        stride = 1
        seq_len = encodings.input_ids.size(1)
        
        running_nlls = []
        
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = 2 
            input_ids = encodings.input_ids[:, 0:end_loc]
            if prev_encodings is not None:
                input_ids = torch.cat((prev_encodings, input_ids), dim=1)
            input_ids = input_ids.to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss


            running_nlls.append(float(neg_log_likelihood.data.cpu().numpy()))

            if end_loc == seq_len:
                nlls_sentences.append(np.mean(running_nlls))
                nlls_full.append(running_nlls)
                if prev_encodings is None:
                    prev_encodings = encodings.input_ids
                else:
                    prev_encodings = torch.cat((prev_encodings, encodings.input_ids), dim=1)
                break

    return nlls_sentences, nlls_full


print('Starting calculating log likelihoods')

train_nlls_sentences = []
train_nlls_full = []
for item in tqdm(X):
    sentences = item.split('_SEP_')[:10]
    nlls_sentences, nlls_full = get_nlls(sentences)
    train_nlls_full.append(nlls_full)
    train_nlls_sentences.append(nlls_sentences)

f = open('ss_nlls-full.txt', 'w')
for x in train_nlls_full:
    f.write(str(x))
    f.write('\n')
f.close()

f = open('ss_nlls-sentences.txt', 'w')
for x in train_nlls_sentences:
    f.write(str(x))
    f.write('\n')
f.close()


