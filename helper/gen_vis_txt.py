import os
import numpy  as np
import pandas  as pd
def read_sim(sim_path):
    return np.load(sim_path)

def read_text_video(text_video_path):
    data = pd.read_csv(text_video_path)
    video_ids = data['video_id'].tolist()
    sentences = data['sentence'].tolist()
    return sentences,video_ids

def sort_str(img_name_list):#caz
    dict_img = {}
    for item in img_name_list:
        # print(item )
        frame_index = item.split('_')[-1].split('.jpg')[0]
        dict_img[item]=int(frame_index)

    dict_img = dict(sorted(dict_img.items(), key=lambda x:x[1],reverse = False))
    sort_img_name_list = list(dict_img.keys())
    return sort_img_name_list

def get_video_frame(tmp_video_ids):
    ret = []
    root_path = "/data/tiankaibin/Text2VideoDataset/MSRVTT/data/MSRVTT/videos/frames"
    pre = "Text2VideoDataset/MSRVTT/data/MSRVTT/videos/frames"
    for tmp_video_id in tmp_video_ids:
        video_path = os.path.join(root_path,tmp_video_id+".mp4")
        frame_names = os.listdir(video_path)
        frame_names = sort_str(frame_names)
        center_frame_path = os.path.join(pre,tmp_video_id+".mp4",frame_names[len(frame_names)//2])
        ret.append(center_frame_path)
    return ret

def get_video(tmp_video_ids):
    ret = []
    root_path = "/data/tiankaibin/Text2VideoDataset/MSRVTT/data/MSRVTT/videos/frames"
    pre = "VideoData"
    for tmp_video_id in tmp_video_ids:
        video_path = os.path.join(root_path,tmp_video_id+".mp4")
        ret.append(os.path.join(pre,tmp_video_id+".mp4"))
    return ret

if __name__=="__main__":
    topk = 5

    sentences,video_ids = read_text_video("/data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv")
    sims = read_sim("/data/tiankaibin/xcliplog/ckpts_dsw/xclip_msrvtt_vit32_supp_eval/sim_matrix.npy")
    f =open("vis.txt","w")
    for i,(sentence,sim) in enumerate(zip(sentences,sims)):
        
        indexs = (np.argsort(sim)[::-1][0:topk]).astype(np.int32)
        # print(sentence,indexs,sim[0:20])
        sim = np.array(sim)
        video_ids = np.array(video_ids)
        high_scores = sim[indexs]
        tmp_video_ids = video_ids[indexs]
        tmp_video_urls = get_video(tmp_video_ids)

        for index,high_score,tmp_video_url,tmp_video_id in zip(indexs,high_scores,tmp_video_urls,tmp_video_ids):
            if index==i:
                s="{}\x02sim:{} GroudTruth\x03{}\x03{}\x01".format(tmp_video_url,high_score,tmp_video_id,sentence)
            else:
                s="{}\x02sim:{}\x03{}\x03{}\x01".format(tmp_video_url,high_score,tmp_video_id,sentence)
            f.write(s)
        
        f.write("\n")
    f.close()        



