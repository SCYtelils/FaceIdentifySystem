# -*- coding:utf-8 -*-
# author TELILS
import cv2

from stu_train import Model
from image_show import show_image
# from stu_picObtain import relight

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cascade_path = r'C:\Users\ASUS\Anaconda3\envs\tensorflow\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'
    model = Model()
    model.load()

    
    i  = 0
    s = 0
    f = 0
    while i<1000:
        _, frame = cap.read()
        
        # 灰度转换
        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 人脸探测器
        cascade = cv2.CascadeClassifier(cascade_path)
        # 获取探测到的面部
        face_rect = cascade.detectMultiScale(frame_grey, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
        print(len(face_rect))
        if len(face_rect) > 0:
            # print('face detected')
            for rect in face_rect:
                x, y = rect[0:2]
                width, height = rect[2:4]
                image = frame[y - 10: y + height, x: x + width]
                # 变更亮度，不同光源下识别
                # image = relight(image,random.uniform(0.5,1.5),random.randint(-50,50))
                result = model.predict(image)
                if result == 0:
                    print('签到成功')
                    s +=1
                    # show_image()
                else:
                    print('签到失败')
                    f+=1

        # 10ms等待键盘输入
        k = cv2.waitKey(100)
        # 按esc键结束
        if k == 27:
            break

        i+=1

    print(i,s,f)
    cap.release()
    cv2.destroyAllWindows()


