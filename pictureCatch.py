import cv2,os,sys,random

out_dir='./my_faces'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# 改变亮度和对比度
def relight(img,alpha=1,bias=0):
    w,h = img.shape[:2]

    for i in range(0,w):
        for j in range(0,h):
            for c in range(3):
                tmp=int(img[i,j,c]*alpha+bias)
                if tmp > 255:
                    tmp=255
                elif tmp<0:
                    tmp=0
                img[i,j,c]=tmp
    
    return img            


