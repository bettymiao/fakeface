import cv2
import os

def getAllName(file_dir, tail_list = ['.jpg']): 
    L=[] 
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in tail_list:
                L.append(os.path.join(root, file))
    return L

face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_alt.xml')


read_dir = "deepfake_celeb"
img_paths = getAllName(read_dir)
print(len(img_paths))

for img_path in img_paths:

    # 检测脸部
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    faces = face_cascade.detectMultiScale(img, 
                            scaleFactor=1.1, 
                            minNeighbors=5, 
                            minSize=(30, 30),
                            flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces)==0:
        continue
    elif len(faces)==1:
        face = faces[0]
    else:
        #多个脸时取最中间的
        
        min_dist = 999
        min_id = -1
        for i,face in enumerate(faces):
            cx = faces[i][0]+faces[i][2]/2
            cy = faces[i][1]+faces[i][3]/2
            dist = ((cx-w/2)**2+(cy-h/2)**2)**0.5
            if dist<min_dist:
                min_dist=dist 
                min_id = i  
        face = faces[i]

    x0,y0,w0,h0 = face 

    crop_x0 = int(max(0,x0-w0*0.3))
    crop_y0 = int(max(0,y0-h0*0.3))
    crop_x1 = int(min(w-1,x0+w0+w0*0.3))
    crop_y1 = int(min(h-1,y0+h0+h0*0.3))
    crop_img = img[crop_y0:crop_y1, crop_x0:crop_x1]

    basename = os.path.basename(img_path)
    save_path = img_path.replace(basename, "crop_"+basename)
    cv2.imwrite(save_path, crop_img)
    # break
    # print('Detected ', len(faces), " face")
    # 在图片中显示检测到的人脸数
    #print(faces) [355 104 248 248]
    # cv2.rectangle(img, (faces[0][0],faces[0][1]), 
    #     (faces[0][0]+faces[0][2], faces[0][1]+faces[0][3]),
    #     (255,0,0),2)
    # # 显示图片
    # cv2.imwrite('img.jpg', img)

