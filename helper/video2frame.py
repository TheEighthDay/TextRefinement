import os, sys
import time
from tqdm import tqdm


import cv2
import logging


logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


def extract_frame(i): 
    
    video_id = all_video_names[i]
    # print('!!!!!!!!')
    # print(video_file)
    # print(video_id )
    
    logger.info('extracting frames from video %d / %d: %s' % (i, num_of_videos, video_id))
    # frame_output_dir = os.path.join(output_dir, videoset[i][-3:], videoset[i])
    # import pdb;pdb.set_trace()
    
    # temp_videoid = video_id.split('_')[0].split('shot')[1]
    # frame_output_dir = os.path.join(output_dir,temp_videoid, videoset[i])
    frame_output_dir = os.path.join(target_floder,all_video_names[i])
    video_input = os.path.join(source_floder,all_video_names[i])


    if not os.path.exists(frame_output_dir):
        os.makedirs(frame_output_dir)

    cap = cv2.VideoCapture(video_input)


    if not cv2.__version__.startswith('2'):
#         length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
    else:
#         length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
#         width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
#     records[video_id] = (fps, length, width, height)

    flag = True
    fcount = 0
    flag, frame = cap.read()
    if(not flag):
        flag, frame = cap.read()

#     print(flag, frame)
    num_of_frame = 0
    while(flag):
        # Write the frame every 0.5 second
        if fcount % fps == 0 or fcount % fps == (fps//2):
        # if fcount % (5*fps) == 0:   # every 5 seconds
            cv2.imwrite(os.path.join(frame_output_dir, '%s_%d.jpg'%(video_id, fcount)), frame)
            num_of_frame += 1
#             total_frame_count += 1
        fcount += 1
        flag, frame = cap.read()
#         print(video_id, flag)

    if fcount > 0:
        print("video id:%s, fps:%s, fcount:%s, num of frames: %s"%(video_id, fps, fcount, num_of_frame))
#         records[video_id] = (fps, length, width, height)
    else:
        logger.error("failed to process %s", video_id)
        failed_video_path = os.path.join(rootpath, 'failed_videos')
        if not os.path.exists(failed_video_path):
            os.makedirs(failed_video_path)
        with open( os.path.join(failed_video_path, video_id)  , 'w') as fw:
            fw.write(video_id)
    
#     del video_file, video_id, frame_output_dir, cap, flag, fcount, frame
    




if __name__ == '__main__':
    rootpath = str(sys.argv[1])  # /data/tiankaibin/Text2VideoDataset/MSRVTT/data/MSRVTT/videos
    video_folder_name =  str(sys.argv[2]) # all3
    frame_folder_name = str(sys.argv[3]) # frames

    source_floder = os.path.join(rootpath,video_folder_name)
    target_floder = os.path.join(rootpath,frame_folder_name)

    if not os.path.exists(target_floder):
        os.makedirs(target_floder)
    all_video_names  = os.listdir(source_floder)

    num_of_videos = len(all_video_names)



    from multiprocessing.pool import Pool
    pool = Pool(processes=40, maxtasksperchild=80) # maxtasksperchild参数用于回收资源，否则会发生内存泄露
    pool.imap_unordered(extract_frame, range(num_of_videos))
    pool.close()
    pool.join() 

