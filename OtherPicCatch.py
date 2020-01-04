import sys,os,cv2

input_path='./faceSource'
output_dir='other_faces'

size=64

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
            