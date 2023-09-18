import os 
import cv2
import random

random.seed(42)




read_dir = "DFMNIST+"
save_dir = "deepfake_dfmnist"
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
fake_dir = os.path.join(read_dir, 'fake_dataset')#10个文件夹，每个1000个视频。
#计划每个文件夹取500视频，每个视频取2帧
subdirs = os.listdir(fake_dir)

print(len(subdirs))
for subdir in subdirs:
    print(subdir)
    try:
        filenames = os.listdir(os.path.join(fake_dir,subdir))
        choose_files = random.sample(filenames, 500)
    except:
        continue # .DS_Store

    for filename in choose_files:
        file_path = os.path.join(fake_dir,subdir,filename)
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
            save_path = os.path.join(save_dir_fake, subdir+"_"+filename[:-4]+"_"+str(i)+".jpg")
            cv2.imwrite(save_path,choose_frames[i])
        videoCapture.release()

### true
true_dir = os.path.join(read_dir, 'real_dataset','selected_train')
filenames = os.listdir(true_dir)
print(len(filenames))
choose_files = random.sample(filenames, 5000)
#selected_train共7000视频，采样5000个，每个取2帧
for filename in choose_files:
    file_path = os.path.join(true_dir,filename)
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
        save_path = os.path.join(save_dir_true, filename[:-4]+"_"+str(i)+".jpg")
        cv2.imwrite(save_path,choose_frames[i])
    videoCapture.release()
