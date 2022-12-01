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

def analyse_length():
    sentences,video_ids = read_text_video("/data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv")
    sims = read_sim("/data/tiankaibin/xcliplog/ckpts_dsw/xclip_msrvtt_vit32_supp_eval/sim_matrix.npy")
    d1 = 0
    d1count = 0

    d2 = 0
    d2count = 0

    d3 = 0
    d3count = 0

    sentencelen =0 

    for i,(sentence,sim) in enumerate(zip(sentences,sims)):
        indexs = (np.argsort(sim)[::-1]).astype(np.int32)
        dis = (np.where(indexs == i)[0])

        # if dis.size == 0:
        #     dis = 20
        # else:
        dis = dis[0]
        
        sentencelen = sentencelen + len(sentence)
        if (len(sentence)<50):
            d1 = d1+dis 
            d1count = d1count+1
        elif (len(sentence)>=50 and len(sentence)<80):
            d2 = d2+dis
            d2count = d2count+1
        else:
            d3 = d3+dis
            d3count = d3count+1
    # print(sentencelen/1000.0) 50
    print("<50: {} {}".format(int(d1/d1count),d1count/1000))
    print(">=50 && <80: {} {}".format(int(d2/d2count),d2count/1000))
    print(">=80: {} {}".format(int(d3/d3count),d3count/1000))

def analyse_bad_case():
    sentences,video_ids = read_text_video("/data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv")
    sims = read_sim("/data/tiankaibin/xcliplog/ckpts_dsw/xclip_msrvtt_vit32_supp_eval/sim_matrix.npy")
    # sentences,video_ids = read_text_video("/data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/rewrite_robot_test.csv")
    # sims = read_sim("/data/tiankaibin/xcliplog/ckpts_dsw/xclip_msrvtt_vit32_supp_eval/sim_matrix_rewrite_robot.npy")

    # sentences,video_ids = read_text_video("/data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/rewrite_tiankaibin_test.csv")
    # sims = read_sim("/data/tiankaibin/xcliplog/ckpts_dsw/xclip_msrvtt_vit32_supp_eval/sim_matrix_rewrite_tiankaibin.npy")

    count_rank200_len50 = 0
    count_rank100_len50 = 0
    count_rank50_len50 = 0
    count_rank10_len50 = 0
    count_rank5_len50 = 0
    count_rank1_len50 = 0

    for i,(sentence,video_id,sim) in enumerate(zip(sentences,video_ids,sims)):
        indexs = (np.argsort(sim)[::-1]).astype(np.int32)
        dis = (np.where(indexs == i)[0])[0]
        if video_id in ["video8912","video7462","video7461","video7619","video7491","video8667","video8938","video7215","video9808","video8442","video8860","video9878"]:
            print("sentnce:{} \n videoid:{} rank:\n{}".format(sentence,video_id,dis))
            print("sentnce:{} \n rank1videoid:{} rank2videoid:{} rank3videoid:{} \n".format(sentence,video_ids[indexs[0]],video_ids[indexs[1]],video_ids[indexs[2]]))


        if(dis>200 and len(sentence)<50):
            # print("{} \n {} \n{}".format(sentence,video_id,dis))
            count_rank200_len50 = count_rank200_len50+1
        if(dis>100 and len(sentence)<50):
            count_rank100_len50 = count_rank100_len50+1
        if(dis>50 and len(sentence)<50):
            count_rank50_len50 = count_rank50_len50+1
        if(dis>10 and len(sentence)<50):
            count_rank10_len50 = count_rank10_len50+1
        if(dis>5 and len(sentence)<50):
            count_rank5_len50 = count_rank5_len50+1
        if(dis>1 and len(sentence)<50):
            count_rank1_len50 = count_rank1_len50+1
        # elif(dis==0):
        #     print("best case:\n {} \n {}".format(sentence,video_id))
    print("rank200_len50:{}".format(count_rank200_len50))
    print("rank100_len50:{}".format(count_rank100_len50))
    print("rank50_len50:{}".format(count_rank50_len50))
    print("rank10_len50:{}".format(count_rank10_len50))
    print("rank5_len50:{}".format(count_rank5_len50))
    print("rank1_len50:{}".format(count_rank1_len50))

def generate_rewrite_sen():
    # rewrite 
    # key,vid_key,video_id,sentence /data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv
    # ret0,msr9770,video9770,a person is connecting something to system
    new_data = pd.read_csv("/data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv")

    # new_data.replace("a family is having coversation","conversation between a family members",inplace=True)
    # new_data.replace("he drew a beautiful picture","he created a lovely image",inplace=True)
    # new_data.replace("cartoons are talking to each otehr","cartoons are communicating with one another",inplace=True)
    # new_data.replace("a person explaining a concept in a show","a person in a program who explains an idea",inplace=True)
    # new_data.replace("music is playing and advertisements was showing","music is playing, while advertising are being shown",inplace=True)
    # new_data.replace("a man is dodging bombs","a guy is avoiding explosives",inplace=True)
    # new_data.replace("person playing a game","person engaged in a game",inplace=True)
    # new_data.replace("a man playing video games","a man who enjoys video games",inplace=True)
    # new_data.replace("a person covers a popular song","a person performs a version of a popular song",inplace=True)
    # new_data.replace("explainin about the scene in the net","explaining the internet scene",inplace=True)
    # new_data.replace("some peole are sitting in hall","some people are seated in the hall",inplace=True)

    # new_data.to_csv("rewrite_robot_test.csv",index=False)
    new_data.replace("a family is having coversation","a couple's family is having coversation",inplace=True)
    new_data.replace("he drew a beautiful picture","he drew a beautiful girl picture ",inplace=True)
    new_data.replace("cartoons are talking to each otehr","a man cartoons are talking to each otehr",inplace=True)
    new_data.replace("a person explaining a concept in a show","a person talking explaining a concept in a show",inplace=True)
    new_data.replace("music is playing and advertisements was showing","music is playing and cartoon advertisements was showing",inplace=True)
    new_data.replace("a man is dodging bombs","a military man is dodging bombs",inplace=True)
    new_data.replace("person playing a game","person playing a airplane game ",inplace=True)
    new_data.replace("a man playing video games","a man playing video games cartoon character",inplace=True)
    new_data.replace("a person covers a popular song","a person covers a popular song about lovers",inplace=True)
    new_data.replace("explainin about the scene in the net","explainin about the game scene in the net",inplace=True)
    new_data.replace("some peole are sitting in hall","some peole are sitting in terrible hall",inplace=True)
    new_data.to_csv("rewrite_tiankaibin_test.csv",index=False)

if __name__=="__main__":
    analyse_length()
    # generate_rewrite_sen()
    analyse_bad_case()

    






