
#畳み込みしてみる

###ライブラリなどの準備
import sys
sys.path.append('..')
import read_data
import glob 
import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers,models
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd 
from keras import backend as K

"""
I tensorflow/core/platform/cpu_feature_guard.cc:142] 
This TensorFlow binary is optimized with oneAPI Deep Neural Network Library 
(oneDNN) to use the following CPU instructions in performance-critical operations:  
AVX2 FMA
"""
#対処法
import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

##In[2]
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold
from tensorflow.keras.utils import plot_model
import scikitplot
from matplotlib import pyplot

##In[3]
# Set Seed
import random
rand = np.random.seed(11)

##In[5]
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
import seaborn as sns
from keras import models
from keras import layers
from keras import optimizers

##In[7]
emotion_label = ["angry","disgust","fear","happy","sad","surprise","neutral"]

face_cascade = cv2.CascadeClassifier('/Users/e185725/practice/fer_practice/ex/opencv_master/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('/Users/e185725/practice/fer_practice/ex/opencv_master/data/haarcascades_cuda/haarcascade_eye.xml')




###モデルの作成
def build_net(optims):
    # Set Seed
    np.random.seed(11)
    random.seed(11)
    net = models.Sequential(name="DCNN")
    net.add(
    Conv2D( #batch size,h,w,d : batch size is no. of img
            filters=64, # change depth(channel) from 1 to 64
            kernel_size=(5,5), #kernel that we use to filter img
    #         input_shape=(img_width, img_height, img_depth),
            activation='elu', #function used to get output of the node
            padding='same', # 
            kernel_initializer='he_normal',
            name='conv2d_1'
        )
    )
    net.add(BatchNormalization(name='batchnorm_1'))
    net.add(
        Conv2D(
            filters=64,
            kernel_size=(5,5),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_2'
        )
    )
    net.add(BatchNormalization(name='batchnorm_2'))
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_1'))
    net.add(Dropout(0.4, name='dropout_1'))
    net.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_3'
        )
    )
    net.add(BatchNormalization(name='batchnorm_3'))
    net.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_4'
        )
    )
    net.add(BatchNormalization(name='batchnorm_4'))
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_2'))
    net.add(Dropout(0.4, name='dropout_2'))
    net.add(
        Conv2D(
            filters=256,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_5'
        )
    )
    net.add(BatchNormalization(name='batchnorm_5'))
    net.add(
        Conv2D(
            filters=256,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_6'
        )
    )
    net.add(BatchNormalization(name='batchnorm_6'))
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_3'))
    net.add(Dropout(0.5, name='dropout_3'))
    net.add(Flatten(name='flatten'))
    net.add(
        Dense(
            128,
            activation='elu',
            kernel_initializer='he_normal',
            name='dense_1'
        )
    )
    net.add(BatchNormalization(name='batchnorm_7'))
    net.add(Dropout(0.6, name='dropout_4')) # 60% to shut the neural and change in order to improve the output


    net.add(
        Dense(7,
            activation='softmax',
            name='out_layer'
        )
    )
    net.compile(
        loss='categorical_crossentropy',
        optimizer=optims,
        metrics=['accuracy']
    )
    net.build((32298, 48, 48, 1))

    net.summary()
    return net

emotion = ["angry","disgust","fear","happy","sad","surprise","neutral"]

checkpoint_path = "./cp_kfc.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

##In[22]
batch_size = 64 #batch size of 32 performs the best.
epochs = 0
optims = [
    optimizers.Nadam(learning_rate=0.0006, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam'),
    optimizers.Adam(0.001),
]


# I tried both `Nadam` and `Adam`, the difference in results is not different but I finally went with Nadam as it is more popular.
net = build_net(optims[1]) 
#print(net)
net.load_weights(checkpoint_path)
model = net
#-----------------------
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("ucOv4OE30aI.mp4")
shift_time = 90
cap.set(0,shift_time*1000)
# cap.set(cv2.CAP_PROP_FPS, 10)


size = (640, 480) # 動画の画面サイズ

fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
writer = cv2.VideoWriter('./output.mp4', fmt, 24.0, size) # ライター作成

while True:
    is_ok,frame = cap.read()
    if not is_ok :break
    frame = cv2.resize(frame,(640,480))
    #frame2 = copy.copy(frame)


    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray,(15,15),0)
    faces = face_cascade.detectMultiScale(gray, 1.1, 6)
    for (x,y,w,h) in faces:
        if (w < 100 or h < 100 and len(faces) == 1):
            continue
        
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        gray_img = gray[x:x+w, y:y+h]
        if ( len(gray_img) == 0):
            continue

        gray_img = cv2.resize(gray_img,dsize=(48,48))
        gray_img = gray_img.reshape((1,48,48,1))
        #print(gray_img.shape)
        predictions = model.predict( gray_img )
        pred = [np.argmax(i) for i in predictions]
        #print(predictions)
        cv2.putText(frame, emotion[pred[0]], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)

    cv2.imshow("video",frame)
    writer.write(frame) 
    key=cv2.waitKey(1)&0xFF


    if key == ord("p"):
        cv2.imwrite("./img/{}/{}.jpg".format( emotion[pred[0]], str( random.randint(0,10000000) ) ),gray_img)
        

    elif key==ord('q'):
        
        break

writer.release() 
cap.release()
cv2.destroyAllWindows()

