import os 
import json
from tqdm import tqdm


if __name__ == "__main__":
    origin_image_caption_file = "OFA/caption_new.txt"
    f = open(origin_image_caption_file,"r")
    origin_image_caption = f.readlines()
    f.close()

    f = open("/data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/MSRVTT_data.json","r")
    listjson = json.load(f)
    f.close()

    
    f = open("/data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/MSRVTT_train.9k.csv","r")
    listcsv = f.readlines()
    f.close()

    train_videos = [x.strip() for x in listcsv[1:]]


    rootpath= "/data/tiankaibin/Text2VideoDataset/MSRVTT/data/MSRVTT/videos/frames/"
    pairs = 200000
    for line in tqdm(origin_image_caption):
        caption = " ".join(line.strip().split(" ")[2:])
        frame = line.strip().split(" ")[1]
        video = frame.split("_")[0]
        new_video = "supp_" + video
        new_frame = "supp_" + frame
        if(not os.path.exists(os.path.join(rootpath,new_video))):
            os.makedirs(os.path.join(rootpath,new_video))
        os.system("cp {} {}".format(os.path.join(rootpath,video,frame),os.path.join(rootpath,new_video,new_frame)))

        if video.split(".")[0] in train_videos:
            listcsv.append(new_video.split(".")[0]+"\n")
            listjson['sentences'].append({"caption":caption,'video_id': new_video.split(".")[0], 'sen_id': pairs})
            pairs = pairs + 1

    
    f = open("/data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/supp_MSRVTT_data.json","w")
    json.dump(listjson,f)
    f.close()

    f = open("/data/tiankaibin/Text2VideoDataset/MSRVTT/msrvtt_data/supp_MSRVTT_train.9k.csv","w")
    f.writelines(listcsv)
    f.close()