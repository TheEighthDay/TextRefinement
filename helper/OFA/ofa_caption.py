import os 
import sys
sys.path.append("/home/lingyun/tiankaibin/phd_video_recognition/Text2Video/helper/OFA/")
from infer import cpation
from tqdm import tqdm

def sort_str(img_name_list):#caz
    dict_img = {}
    for item in img_name_list:
        # print(item )
        frame_index = item.split('_')[-1].split('.jpg')[0]
        dict_img[item]=int(frame_index)

    dict_img = dict(sorted(dict_img.items(), key=lambda x:x[1],reverse = False))
    sort_img_name_list = list(dict_img.keys())
    return sort_img_name_list

if __name__ == '__main__':
    rootpath = str(sys.argv[1])  # /data/tiankaibin/Text2VideoDataset/MSRVTT/data/MSRVTT/videos
    frame_folder_name = str(sys.argv[2]) # video
    #video_names = ["video8888.mp4","video8255.mp4","video4282.mp4","video8169.mp4","video7383.mp4","video727.mp4","video397.mp4","video9970.mp4"]


    video_names = os.listdir(os.path.join(rootpath,frame_folder_name))

    f = open("caption.txt","w")

    for video_name in tqdm(video_names):
        frame_names = os.listdir(os.path.join(rootpath,frame_folder_name,video_name))
        sorted_frame_names = sort_str(frame_names)
        try:
            mid_frame_name = sorted_frame_names[len(sorted_frame_names)//2]
            result = cpation(os.path.join(rootpath,frame_folder_name,video_name,mid_frame_name))
        except:
            mid_frame_name = sorted_frame_names[len(sorted_frame_names)//2+1]
            result = cpation(os.path.join(rootpath,frame_folder_name,video_name,mid_frame_name))
        f.write("{} {} {}\n".format(video_name,mid_frame_name,result))
    f.close()




    