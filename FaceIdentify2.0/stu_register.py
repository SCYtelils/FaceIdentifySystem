# -*- coding:utf-8 -*-
import cv2, os

from datetime import datetime
from stu_train import Model
from image_show import show_image

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cascade_path = r'C:\Users\ASUS\Anaconda3\envs\tensorflow\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'
    model = Model()
    model.load()
    face_rect=[]
    count = 0
    result = 0
    while count<10 :
        # while True:
        _, frame = cap.read()

        # 灰度转换
        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 人脸探测器
        cascade = cv2.CascadeClassifier(cascade_path)
        # 获取探测到的面部
    
        face_rect = cascade.detectMultiScale(frame_grey, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
        print('第%d次录入'%(count+1))
        if len(face_rect) > 0:
            print('face detected')
            for rect in face_rect:
                x, y = rect[0:2]
                width, height = rect[2:4]
                image = frame[y - 10: y + height, x: x + width]
                result_u = model.predict(image)
                if result_u == 0:
                    print('录入成功')
                    result = result + 1
                else:
                    print('录入失败')
            count += 1
                
                    
    if result >= 9:
        print('签到成功')
        f = open(r'C:\Users\ASUS\git\FaceIdentifySystem\FaceIdentify2.0\result.txt','w')
        time = datetime.now()
        record = 'regist success :' + str(time)
        f.write(record)
        # show_image()
    else:
        print('签到失败')

        # # 10ms等待键盘输入
        # k = cv2.waitKey(100)
        # # 按esc键结束
        # if k == 27:
        #     break

cap.release()
cv2.destroyAllWindows()


