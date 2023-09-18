import os 
import cv2
import random

random.seed(42)




read_dir = "./"
save_dir = "deepfake_celeb"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_dir_fake = os.path.join(save_dir,'fake')
if not os.path.exists(save_dir_fake):
    os.makedirs(save_dir_fake)
save_dir_true = os.path.join(save_dir,'true')
if not os.path.exists(save_dir_true):
    os.makedirs(save_dir_true)
count = 10000

### fake
fake_dir = os.path.join(read_dir, 'Celeb-synthesis')#5639个文件，采样5000个，每个视频取2帧
filenames = os.listdir(fake_dir)
print(len(filenames))
choose_files = random.sample(filenames, 5000)
#selected_train共7000视频，采样5000个，每个取2帧
for filename in choose_files:
    file_path = os.path.join(fake_dir,filename)
    #print(file_path)
    videoCapture = cv2.VideoCapture(file_path)
    frames = []
    success, frame = videoCapture.read()
    while success :
        frames.append(frame)
        success, frame = videoCapture.read()
    #print(len(frames))
    choose_frames = random.sample(frames,2)
    for i in range(len(choose_frames)):
        save_path = os.path.join(save_dir_fake, filename[:-4]+"_"+str(i)+".jpg")
        cv2.imwrite(save_path,choose_frames[i])
    videoCapture.release()


### true
true_dir = os.path.join(read_dir, 'Celeb-real')
filenames = os.listdir(true_dir)
#selected_train共890视频，每个取11帧 =9790
for filename in filenames:
    file_path = os.path.join(true_dir,filename)
    #print(file_path)
    videoCapture = cv2.VideoCapture(file_path)
    fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

    frames = []
    success, frame = videoCapture.read()
    while success :
        frames.append(frame)
        success, frame = videoCapture.read()
    #print(len(frames))
    choose_frames = random.sample(frames,min(fNUMS,11))
    for i in range(len(choose_frames)):
        save_path = os.path.join(save_dir_true, filename[:-4]+"_"+str(i)+".jpg")
        cv2.imwrite(save_path,choose_frames[i])
    videoCapture.release()
