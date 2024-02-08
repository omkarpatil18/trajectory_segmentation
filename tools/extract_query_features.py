import torch
import os.path
from tqdm import tqdm
import jsonlines
import numpy as np
from transformers import AutoTokenizer, CLIPTextModel

model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def extract_features (qid, query):

    path = '/home/local/ASUAD/draj5/dataset/features/clip_text_features/'
    if not os.path.exists(path):
        os.makedirs(path)

    if os.path.isfile (f"{path}qid{qid}.npz"):
        return
    inputs = tokenizer (query, padding = True, return_tensors = "pt")
    outputs = model (**inputs)
    lsh = torch.squeeze (outputs.last_hidden_state.detach (), 0)
    po = torch.squeeze (outputs.pooler_output.detach (), 0)
    #print (lsh.shape, po.shape)
    np.savez (f"{path}qid{qid}", last_hidden_state = lsh, pooler_output = po)

def extract_files (dir_name):
    files = os.listdir (dir_name)
    for name in tqdm (files, desc = "Extracting Text Features"):
        reader = jsonlines.open (f"{dir_name}{name}")
        for data in reader:
            extract_features (data['qid'], data['query'])

if __name__ == '__main__':
    extract_files ('/home/local/ASUAD/draj5/dataset/json/')
