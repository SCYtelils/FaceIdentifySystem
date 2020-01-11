#-*- coding: UTF-8 -*-
import os,cv2
import numpy as np

IMAGE_SIZE = 64

# 格式化图片
def resize_with_pad(image, height=IMAGE_SIZE, width=IMAGE_SIZE):

    # 计算图片需要补全区域的大小
    def get_padding_size(image):
        h, w, _=image.shape
        longest_edge = max(h,w)
        top, bottom, left, right = (0,0,0,0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right
    
    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    # 用黑色填充64*64的缺失区域
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    resized_image = cv2.resize(constant, (height, width))

    return resized_image

images = []
labels = []

# 获取图片信息，对信息进行标记和预处理
def traverse_dir(path):
    
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        print(abs_path)
        if os.path.isdir(abs_path):  # dir
            traverse_dir(abs_path)
        else:                        # file
            if file_or_dir.endswith('.jpg'):
                image = read_image(abs_path)
                images.append(image)
                labels.append(path)

    return images, labels

def read_image(file_path):
    image = cv2.imread(file_path)
    image = resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)

    return image 

def extract_data(path):
    images, labels = traverse_dir(path)
    images = np.array(images)
    labels = np.array([0 if label.endswith('stuFace') else 1 for label in labels])

    return images, labels
