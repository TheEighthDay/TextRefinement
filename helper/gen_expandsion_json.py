import os 
import json
from tqdm import tqdm


def generate_new_query_json_for_msrvtt(N=3,replace=True):
    """
    N:一般我会对于训练集所有句子生成10倍的扩充,可以选择其中几份扩充原先的query,不要超过10,thanks。
    """
    origin_video_caption_file = "/home/lingyun/tiankaibin/phd_video_recognition/Text2Video/X-CLIP/ckpts_dsw/xclipbridge_msrvtt_vit32_recover1masktoken/msrvtt_9k.txt" #xclipbridge_msrvtt_vit32_recover1saliencymasktoken
    f = open(origin_video_caption_file,"r")
    origin_video_caption = f.readlines()
    f.close()

    origin_video_caption = origin_video_caption[0:int(180000*N)]

    f = open("/data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/MSRVTT_data.json","r")
    listjson = json.load(f)
    f.close()
    if replace:
        listjson['sentences'] = []
        pairs = 0
    else:
        pairs = 200000
    
    for line in tqdm(origin_video_caption):
        caption = " ".join(line.strip().split(" ")[1:])
        caption = caption.replace("<|startoftext|>","")
        caption = caption.replace("<|endoftext|>","")
        caption = caption.strip()
        video = line.strip().split(" ")[0]
        listjson['sentences'].append({"caption":caption,'video_id': video, 'sen_id': pairs})
        pairs = pairs + 1

    f = open("/data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/msrvttpretrain_mlmrecover1token_9k_MSRVTT_data_less_replace.json","w")
    json.dump(listjson,f)
    f.close()

if __name__ == "__main__":
    generate_new_query_json_for_msrvtt(N=1)
    #msrvttpretrain_mlm_9k_MSRVTT_data_replace_less.json
    #msrvttpretrain_mlm1saliencytoken_9k_MSRVTT_data_replace_less.json
    #msrvttpretrain_mlm5saliencytoken_9k_MSRVTT_data_replace_less.json
    #msrvttpretrain_mlm5token_9k_MSRVTT_data_replace_less.json
    #msrvttpretrain_mlmrecover1token_9k_MSRVTT_data_less_replace.json
    #msrvttpretrain_mlmrecover1saliencytokentoken_9k_MSRVTT_data_less_replace.json