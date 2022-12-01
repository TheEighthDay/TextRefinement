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
# cla_probs.npy (1000, 9648)
# cross_atten_matrix.npy  (1000, 12, 31, 49)
# sim_matrix.npy (1000, 1000)

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

def get_word_vocab():
    word_vocab = tokenizer.vocab
    # concept_word_vocab_path = "/home/lingyun/tiankaibin/phd_video_recognition/Text2Video/X-CLIP/dataloaders/msrvtt_word_vocab.txt"
    # f = open(concept_word_vocab_path,"r")
    # datalines = f.readlines()
    # f.close()
    # word_vocab = {}
    # word_concept = 0
    # word_vocab_pre = []
    # for dataline in datalines:
    #     word = dataline.strip().split(" ")[0]
    #     word_vocab_pre.append(word)
    
    # allstopwords = stopwords.words('english') 
    # for i in range(len(allstopwords)):
    #     allstopwords[i] = allstopwords[i]+"</w>"
    
    # words = tokenizer.tokenize(" ".join(word_vocab_pre))
    # words = [word for word in words if word not in allstopwords]

    # for word in words:
    #     if word not in word_vocab:
    #         word_vocab[word] = word_concept
    #         word_concept = word_concept + 1
    # print(len(word_vocab))
    return word_vocab


def get_frame(video_id):
    video, video_mask = msrvtt_testset._get_rawvideo([video_id])
    # print(video_mask)
    # print(video.shape)
    a,b,c,d,e,f = video.shape
    video = video.reshape((b,d,e,f))
    return video

def get_word(sentence):
    words = tokenizer.tokenize(sentence)
    words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
    total_length_with_CLS = 31
    if len(words) > total_length_with_CLS:
        words = words[:total_length_with_CLS]
    words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]
    words = words[1:] #去除CLS_TOKEN,SEP_TOKEN
    # print(len(words),words)
    return words


def plot_heatmap(sentence,word_in_sentence,frame,attention_map_blocks,save_root):
    save_path = os.path.join(save_root,word_in_sentence+".png")
    frame = torch.tensor(frame)
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    for i in range(3):
        frame[i] = frame[i] * std[i] + mean[i]
    frame = frame * 255
    frame = np.uint8(np.transpose(frame, (1, 2, 0)))
    frame = frame[:,:,[2,1,0]]

    # attention_map = attention_map_blocks.reshape((7,7)).repeat(32, axis = 0).repeat(32, axis = 1)
    attention_map_blocks = attention_map_blocks.reshape((7,7))
    attention_map = torch.nn.functional.interpolate(torch.tensor([[attention_map_blocks]]), size=(224,224),mode="bilinear")
    attention_map = attention_map.numpy().reshape((224,224))
    heatmap = np.uint8(255 * attention_map)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    frame = np.uint8(0.7*frame+0.3*heatmap)
    cv2.imwrite(save_path,frame)


def read_sentence_video_pair():
    pair_file_path = "/data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv"
    data = pd.read_csv(pair_file_path)
    video_ids = data['video_id'].tolist()
    sentences = data['sentence'].tolist()

    video_frames = []
    sentence_words = []


    for video_id in tqdm(video_ids):
        frames = get_frame(video_id)
        video_frames.append(frames)

    for sentence in tqdm(sentences):
        words = get_word(sentence)
        sentence_words.append(words)

    return video_ids,sentences,list(zip(sentence_words,video_frames))
        


def generate_heatmap():
    save_root = "/home/lingyun/tiankaibin/phd_video_recognition/Text2Video/X-CLIP/heatmap"
    root_dir = "/home/lingyun/tiankaibin/phd_video_recognition/Text2Video/X-CLIP/ckpts_dsw/xclipbridge_msrvtt_vit32_mlm_eval/"
    attention_map_path = os.path.join(root_dir,"attens.npy")
    #(1000,8 , 12, 31, 49)
    attention_map = np.load(attention_map_path)
    attention_map = attention_map.mean(1)
    #(1000, 12, 31, 49)
    # print(attention_map.shape)
    video_ids,sentences,words_frames_pairs = read_sentence_video_pair()

    for k,(words,frames) in tqdm(enumerate(words_frames_pairs)):
        
        for i in range(len(words)):
            # print(words[i])
            max_value = attention_map[k,:,i,:].max() #[k,:,i,:] 对每个词的atten 归一化
            min_value = attention_map[k,:,i,:].min() #[k,:,:,:] 对所有词的atten 归一化

            for j in range(len(frames)):
                # print(frames[j])
                
                # words[i] = words[i].replace("</w>","")
                # print(attention_map[k][j][i])
                attention_map[k][j][i] = (attention_map[k][j][i]-min_value)/(max_value-min_value)
                if not os.path.exists(os.path.join(save_root,video_ids[k],"frame_{}".format(j))):
                    os.makedirs(os.path.join(save_root,video_ids[k],"frame_{}".format(j)))

                # print(attention_map[k][j][i])
                
                plot_heatmap(sentences[k],words[i],frames[j],attention_map[k][j][i],os.path.join(save_root,video_ids[k],"frame_{}".format(j)))

    # #(length_pair, frames, word_tokens, patch_tokens)
    # #(1000, 12, 31, 49)
    # #归一化
def check_cls_effect():
    pair_file_path = "/data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv"
    data = pd.read_csv(pair_file_path)
    video_ids = data['video_id'].tolist()
    sentences = data['sentence'].tolist()[0:10]

    word_vocab = get_word_vocab()
    word_vocab_reverse = dict()

    for k,v in word_vocab.items():
        word_vocab_reverse[v] = k
    root_dir = "/home/lingyun/tiankaibin/phd_video_recognition/Text2Video/X-CLIP/ckpts_dsw/xclipbridge_msrvtt_vit32_concept_stopwords_infer/"
    cls_probs = np.load(os.path.join(root_dir,"cla_probs.npy"))
    cls_probs_index = np.argsort(cls_probs,axis=1)[:,::-1]

    predict_words=[]

    for cls_prob_index in cls_probs_index[:10]:
        cls_prob_index_rank10 = cls_prob_index[0:10]
        tmp = []
        for x in cls_prob_index_rank10:
            tmp.append(word_vocab_reverse[x])
        tmp = [str(x) for x in tmp]
        tmp = " ".join(tmp)
        predict_words.append(tmp)
    
    for p,r in zip(predict_words,sentences):
        print("-----------")
        print(p)
        print(r)




if __name__=="__main__":
    generate_heatmap()
    # check_cls_effect()

    #"<|startoftext|>","<|endoftext|>"  : 49406, 49407  
    # total 49408
    # mask "!!!!!!!!!!!</w>" id: 49339
    # pad "!" id: 0
