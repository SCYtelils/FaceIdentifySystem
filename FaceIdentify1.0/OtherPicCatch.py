import sys,os,cv2

input_path=r'C:\Users\ASUS\git\FaceIdentifySystem\FaceIdentify2.0\data\otherFace'
output_dir=r'C:\Users\ASUS\git\FaceIdentifySystem\FaceIdentify2.0\data\otherFace'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取分类器
haar = cv2.CascadeClassifier(r'C:\Users\ASUS\Anaconda3\envs\tensorflow\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

index=1
# 返回路径，文件夹名称，文件名称
for (path,dirnames,filenames) in os.walk(input_path):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('being processed picture %s'% str(index))
            img_path=path+'/'+filename
        
        # 从文件中读取图片
        img=cv2.imread(img_path)
        # 转为灰度图片
        gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # 使用haar进行人脸检测
        faces=haar.detectMultiScale(gray_img,1.3,5)
        for f_x,f_y,f_w,f_h in faces:
            face = img[f_y:f_y+f_h,f_x:f_x+f_w]
            # 统一保存为64*64格式
            face = cv2.resize(face,(64,64))
            cv2.imshow('img',face)
            cv2.imwrite(output_dir+'/'+'oth'+str(index)+'.jpg',face)
            index = index + 1
        if index == 10000:
            sys.exit(0)
            key=cv2.waitKey(30)&0xff
            if key == 27:
                sys.exit(0)
