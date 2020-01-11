#-*- coding: UTF-8 -*-
import cv2,os
from PIL import Image 

name = 'stu'
pic_obtain_num = 10000   # 获取照片的数量
out_path = r'D:\FaceData\data\otherFace'

def get_face(path=None):
    cap = cv2.VideoCapture(path)
    # cap = cv2.VideoCapture(0)
    classfier = cv2.CascadeClassifier(r'C:\Users\ASUS\Anaconda3\envs\tensorflow\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
    cap_check = cap.isOpened()
    frame_count = 0
    out_count = 0
    while cap_check:
        frame_count += 1
        if out_count > pic_obtain_num:
            break
        cap_check, frame = cap.read()  # 读取一帧
        
        
        params = []
        params.append(2)
        # 将图片转换成灰度图片
        grey_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        # 获取脸部位置
        face_rects = classfier.detectMultiScale(grey_img, scaleFactor = 1.3, minNeighbors = 5)
        if len(face_rects) > 0:
            for face in face_rects:
                x, y, w, h = face
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                

                # 将预处理的图片存到目标路径下
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                cv2.imwrite(out_path+'\\'+name+'%d.jpg' % out_count,image)
                out_count += 1
                print('成功获取学生'+name+'的第%d个面部图片'%out_count)
    cap.release()
    cv2.destroyAllWindows()
    print('总帧数:', frame_count)
    print('提取脸部:',out_count)

if __name__ == "__main__":
    get_face(0) # 0表示本机摄像头


