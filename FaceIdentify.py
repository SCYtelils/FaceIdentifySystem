import tensorflow as tf
import sys,os,random,cv2
import numpy as np
from sklearn.model_selection import train_test_split

my_faces_path='my_faces'
other_faces_path='other_faces'
size=64

imgs=[]
labs=[]

def getPaddingSize(img):
    h,w,_=img.shape
    top,bottom,left,right=(0,0,0,0)
    longest =max(h,w)

    if w<longest:
        tmp=longest-w

        left=tmp//2
        right=tmp-left
    elif h<longest:
        tmp=longest-h
        top=tmp//2
        bottom=tmp-top
    else:
        pass
    return top,bottom,left,right

# 读取处理好的图片
def readData(path,h=size,w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path+'/'+filename
            img=cv2.imread(filename)
            top,bottom,left,right=getPaddingSize(img)
        # 将图片放大，扩充图片边缘部分
        img=cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0,0,0])
        img=cv2.resize(img,(h,w))
        imgs.append(img)
        # 添加标签，根据路径判别类别
        if path==my_faces_path:
            labs.append([0,1])
        else:
            labs.append([1,0])
    
readData(my_faces_path)
readData(other_faces_path)

# 将图片数据和标签转换成数组
imgs=np.array(imgs)
# 随机划分测试集和训练集
train_x,test_x,train_y,test_y = train_test_split(imgs,labs,test_size=0.25,random_state=random.randint(0,100))
# 参数  图片数据的总数 图片的高，宽，通道
train_x=train_x.reshape(train_x.shape[0],size,size,3)
test_x=test_x.reshape(test_x.shape[0],size,size,3)
            
# 将数据转换成小于1的数
train_x=train_x.astype('float32')/255.0
test_x=test_x.astype('float32')/255.0

print('train size: %s ,test size:%s' % (len(train_x),len(test_x)))


#分批次
batch_size=128
num_batch=len(train_x)//batch_size

x=tf.placeholder(tf.float32,[None,size,size,3])
y_=tf.placeholder(tf.float32,[None,2])

keep_prob_5=tf.placeholder(tf.float32)
keep_prob_75=tf.placeholder(tf.float32)

#随机权值向量
def weightVariable(shape):
    init = tf.random_normal(shape,stddev=0.01)
    return tf.Variable(init)

#随机偏置向量
def baisVariable(shape):
    init=tf.random_normal(shape)
    return tf.Variable(init)

#定义卷积函数
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#定义最大池化
def maxPool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#定义丢失函数
def dropout(x,keep):
    return tf.nn.dropout(x,keep)

def cnnLayer():
    #第一层
    #卷积核大小（3,3） 通道是3 输出通道32
    W1=weightVariable([3,3,3,32])
    b1=baisVariable([32])

    #卷积
    conv1=tf.nn.relu(conv2d(x,W1)+b1)
    #池化
    pool1=maxPool(conv1)
    #减少过拟合，随机让某些权值不更新
    drop1=dropout(pool1,keep_prob_5)

    #第二层
    W2=weightVariable([3,3,32,64])
    b2=baisVariable([64])
    conv2=tf.nn.relu(conv2d(drop1,W2)+b2)
    pool2=maxPool(conv2)
    drop2=dropout(pool2,keep_prob_5)

    #第三层
    W3=weightVariable([3,3,64,64])
    b3=baisVariable([64])
    conv3=tf.nn.relu(conv2d(drop2,W3)+b3)
    pool3=maxPool(conv3)
    drop3=dropout(pool3,keep_prob_5)

    #全连接层
    Wf=weightVariable([64*64,512])
    bf=baisVariable([512])
    #将特征图展开
    drop3_flat=tf.reshape(drop3,[-1,8*8*64])
    dense=tf.nn.relu(tf.matmul(drop3_flat,Wf)+bf)
    dropf=dropout(dense,keep_prob_75)

    #输出层
    Wout=weightVariable([512,2])
    bout=weightVariable([2])

    out=tf.add(tf.matmul(dropf,Wout),bout)
    return out

output=cnnLayer()
predict=tf.argmax(output,1)

saver=tf.train.Saver()
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

# 判断是否为我的脸 根据标签结尾的0，1判断是否为预测结果
def is_my_face(image):
    res=sess.run(predict,feed_dict={x:[image/255.0],keep_prob_5:1.0,keep_prob_75:1.0})
    if res[0]==1:
        return True
    else:
        return False

# 使用OpenCV检测人脸
haar = cv2.CascadeClassifier(r'C:\Users\ASUS\Anaconda3\envs\tensorflow\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

camera=cv2.VideoCapture(0)
flag=1
trance=1

while True:
    _,img=camera.read()
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=haar.detectMultiScale(gray_img,1.3,5)

    if not len(faces):
        cv2.imshow('img',img)
        key=cv2.waitKey(30)&0xff
        if key==27:
            sys.exit(0)
    
    # 标记矩形框
    for f_x,f_y,f_w,f_h in faces:
        face = img[f_y:f_h+f_y,f_x:f_x+f_w]
        face = cv2.resize(face,(size,size))
        flag = flag +1

        if is_my_face(face):
            trance += 1
            print('检测到主人啦！(@-_-@)+%s'%trance)
            cv2.imwrite('test_faces/'+str(flag)+'.jpg',face)
            cv2.rectangle(img,(f_x,f_y),(f_x+f_w,f_y+f_h),(0,0,255),3)
        else:
            print('没有主人!T_T+%s'%trance)
            cv2.rectangle(img,(f_x,f_y),(f_x+f_w,f_y+f_h),(255,0,0),3)
            cv2.imshow('image',img)
            key=cv2.waitKey(30)&0xff
            if key==27:
                sys.exit(0)
# saver.restore(sess,tf.train.latest_checkpoint('C:/Users/ASUS/git/tmp/'))

sess.close()