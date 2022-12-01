import sys
sys.path.append("/home/lingyun/tiankaibin/phd_video_recognition/Text2Video/X-CLIP")
import numpy as np 
import os
import pandas as pd
from nltk.corpus import stopwords
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader
from tqdm import tqdm
import numpy as np
import cv2
import torch
# cla_probs.npy (1000, 9648)
# cross_atten_matrix.npy  (1000, 12, 31, 49)
# sim_matrix.npy (1000, 1000)

tokenizer = ClipTokenizer()
vab = tokenizer.vocab
vab_reverse = {}
for k,v in vab.items():
    vab_reverse[v] = k

SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>","MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

def get_word(sentence):
    words = tokenizer.tokenize(sentence)
    words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
    total_length_with_CLS = 31
    if len(words) > total_length_with_CLS:
        words = words[:total_length_with_CLS]
    words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]
    # words = words[1:-1] #去除CLS_TOKEN,SEP_TOKEN
    # print(len(words),words)
    return words


def vis_sentence_mask_predict():
    pair_file_path = "/data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv"
    data = pd.read_csv(pair_file_path)
    video_ids = data['video_id'].tolist()
    sentences = data['sentence'].tolist()

    mask_dir_path = "/home/lingyun/tiankaibin/phd_video_recognition/Text2Video/X-CLIP/ckpts_dsw/xclipbridge_msrvtt_vit32_mlm"

    mask_labels_path = os.path.join(mask_dir_path,"mask_labels.npy")
    mask_probs_path = os.path.join(mask_dir_path,"mask_probs.npy")
    mask_labels = np.load(mask_labels_path)
    mask_probs = np.load(mask_probs_path)
    mask_preds = np.argmax(mask_probs,2)

    B,word_size = mask_preds.shape
    f =open("mask_token_vis_result.txt","w")
    for i in range(B):
        sen = sentences[i]
        video_id = video_ids[i]

        origin_words = get_word(sen)
        mask_words=[]
        predicted_words=[]
        for j in range(word_size):
            if mask_labels[i,j]!=-100:
                mask_words.append(vab_reverse[mask_labels[i,j]])
                predicted_words.append(vab_reverse[mask_preds[i,j]])
        f.write("---------------{}----------------\n".format(video_id))
        f.write(" ".join(origin_words)+"\n")
        f.write(" ".join(mask_words)+"\n")
        f.write(" ".join(predicted_words)+"\n")
    f.close()




    # print(mask_labels.shape)
    # print(mask_probs.shape)

if __name__=="__main__":
    vis_sentence_mask_predict()

