import sys
sys.path.append("/home/lingyun/tiankaibin/phd_video_recognition/Text2Video/X-CLIP")
import numpy as np 
import os
import pandas as pd
from nltk.corpus import stopwords
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from dataloaders.dataloader_msrvtt_retrieval_mlm import MSRVTT_DataLoader_MLM
from tqdm import tqdm
import numpy as np
import cv2
import torch
import json

# cla_probs.npy (1000, 9648)
# cross_atten_matrix.npy  (1000, 12, 31, 49)
# sim_matrix.npy (1000, 1000)
import nltk
tokenizer = ClipTokenizer()
msrvtt_testset = MSRVTT_DataLoader_MLM(
        subset="test_mlm",
        csv_path="/data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv",
        features_path="/data/tiankaibin/Text2VideoDataset/MSRVTT/data/MSRVTT/videos/frames",
        max_words=32,
        feature_framerate=1,
        tokenizer=tokenizer,
        max_frames=12,
        frame_order=0,
        slice_framepos=2,
    )
SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>","MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
tags = set(['NN','NNS','NNP','NNPS', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RP', 'RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS'])

vocab = tokenizer.vocab

# print(vocab["a</w>"])
pos_vocab = {}



count = 0
for i,k in enumerate(vocab.keys()):
    word = k
    if word not in pos_vocab:
        word_strip = word.replace("</w>","")
        pos_tag = nltk.pos_tag([word_strip])[0][1]
        if pos_tag in tags:
            count = count+1
            pos_vocab[word] = 1
        else:
            pos_vocab[word] = 0
        
print(count,len(vocab)) #去掉了800个停用词
#49091 49408
print(pos_vocab["a"])
print(pos_vocab["a</w>"])
f = open("/data/tiankaibin/Text2VideoDataset/pos.json","w")
json.dump(pos_vocab,f)
f.close()
