# -*- coding:utf-8 -*-
import cv2

from stu_train import Model
from image_show import show_image

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cascade_path = r'C:\Users\ASUS\Anaconda3\envs\tensorflow\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'
    model = Model()
    model.load()
    while True:
        _, frame = cap.read()

        # 灰度转换
        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 人脸探测器
        cascade = cv2.CascadeClassifier(cascade_path)
        # 获取探测到的面部
        face_rect = cascade.detectMultiScale(frame_grey, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
        
        if len(face_rect) > 0:
            print('face detected')
            color = (255,255,255)  # 白
            for rect in face_rect:
                x, y = rect[0:2]
                width, height = rect[2:4]
                image = frame[y - 10: y + height, x: x + width]

                result = model.predict(image)
                if result == 0:
                    print('检测到目标对象')
                    show_image()
                else:
                    print('非目标对象')

        # 10ms等待键盘输入
        k = cv2.waitKey(100)
        # 按esc键结束
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


