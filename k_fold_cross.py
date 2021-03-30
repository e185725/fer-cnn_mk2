
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

##In[9][10]~[15]
#pixels => data
#テストデータと訓練データを格納するための配列を用意する
train_data,train_label = read_data.read_data( read_data.train_name )
test_data ,test_label = read_data.read_data( read_data.test_file )

#データを学習できるように整形
train_data,test_data = train_data/255.0,test_data/255.0
train_data = train_data.reshape((28708,48,48,1))
test_data = test_data.reshape((3589,48,48,1))
test_label = to_categorical(test_label)
train_label = to_categorical(train_label)

X_train, X_valid, y_train, y_valid = train_data,test_data,train_label,test_label

print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

##In[16]
"""
plt.figure(0, figsize=(12,6))
# for i in range(1, 13):
#     plt.subplot(3,4,i)
#     plt.imshow(train_pixels[i, :, :, 0], cmap="gray")
plt.imshow(train_data[0,:48,:48,0], cmap="gray")
plt.show()
"""

##In[17]

##In[18]
# Model configuration
batch_size = 32
loss_function = "categorical_crossentropy"
no_classes = 7 # 7 emotions
no_epochs = 50
optims = optimizers.Nadam(learning_rate=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam')
optimizer = optims
verbosity = 1
num_folds = 10
input_shape = (48, 48, 1)

##In[19]
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

##In[20]
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.00005,
    patience=11,
    verbose=1,
    restore_best_weights=True,
)

lr_scheduler = ReduceLROnPlateau( #Reduce learning rate when a metric has stopped improving.
    monitor='val_accuracy',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1,
)

callbacks = [
#     early_stopping,
    lr_scheduler,
]

checkpoint_path = "./cp_kfc.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# チェックポイントコールバックを作る
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)

##In[21]
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
)
train_datagen.fit(X_train)

##In[22]
batch_size = 64 #batch size of 32 performs the best.
epochs = 0
optims = [
    optimizers.Nadam(learning_rate=0.0006, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam'),
    optimizers.Adam(0.001),
]


# I tried both `Nadam` and `Adam`, the difference in results is not different but I finally went with Nadam as it is more popular.
net = build_net(optims[1]) 
print(net)
net.load_weights(checkpoint_path)
history = net.fit(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_valid, y_valid),
    steps_per_epoch=len(X_train) / batch_size,
    epochs=epochs,
    callbacks=[callbacks,cp_callback],
    #use_multiprocessing=True
)


test_loss,test_acc = net.evaluate(X_valid,y_valid,verbose=2)
print(test_acc)

###ヒートマップの表示と保存

y = [np.argmax(i) for i in y_valid]
predictions = net.predict(X_valid)
emotion = ["angry","disgust","fear","happy","sad","surprise","neutral"]
pred = [np.argmax(i) for i in predictions]
cm = confusion_matrix(y, pred)
test_len = np.array([[467],[56],[496],[895],[653],[415],[607]])
cm = cm / test_len
cm = np.round(cm,3)

cm = pd.DataFrame(data=cm, index=emotion, 
                           columns= emotion)

sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='Blues',fmt="g")
plt.xlabel("Pre", fontsize=13)
plt.ylabel("True", fontsize=13)
plt.show()
plt.savefig('sklearn_confusion_matrix.png')


##In[22]
# acc = history.history["accuracy"]
# val_acc = history.history["val_accuracy"]
# loss = history.history["loss"]
# val_loss = history.history["val_loss"]

# epochs = range(1, len(acc) + 1)

# plt.plot(epochs, acc, "g", label = "Training acc")
# plt.plot(epochs, val_acc, "r", label = "Validation acc")
# plt.title("Accuracy")
# plt.legend()
# plt.figure()

# plt.plot(epochs, loss, "g", label = "Training Loss")
# plt.plot(epochs, val_loss, "r", label = "Validation Loss")
# plt.title("Loss")

# plt.legend()
# plt.show()

#net.save('./my_facial_expression_reg_0.6')

###重みの可視化
# def plot_conv_weights(filters):
#     filter_num = filters.shape[3]
#     try:
#         for i in range(filter_num):
#             plt.subplot(filter_num//6 + 1, 6, i+1)
#             plt.xticks([])
#             plt.yticks([])
#             plt.xlabel(f'filter {i}')
#             plt.imshow(filters[:, :, 0, i])
#         plt.show()
#         plt.clf()
#         plt.close()
#     except:
#         pass

# # 1層目 (Conv2D)
# plot_conv_weights(net.get_layer(name="conv2d_1").get_weights()[0])
# # 2層目 
# plot_conv_weights(net.get_layer(name="conv2d_1").get_weights()[0])
# # 3層目 
# plot_conv_weights(net.get_layer(name="conv2d_2").get_weights()[0])


###フィルターの可視化
# 畳み込み層のみを抽出
# model = net 
# conv_layers = [l.output for l in model.layers]
# conv_model = models.Model(inputs=model.inputs, outputs=conv_layers)
# # 畳み込み層の出力を取得
# conv_outputs = conv_model.predict(test_data[:15])

# for i in range(len(conv_outputs)):
#     print(f'layer {i}:{conv_outputs[i].shape}')

# def plot_conv_outputs(outputs):
#     filters = outputs.shape[2]#画像選択
#     plt.figure(figsize=(7,7))
#     for i in range(filters):
#         plt.subplot(filters//8 + 1, 8, i+1)
#         plt.xticks([])
#         plt.yticks([])
#         #plt.xlabel(f'filter {i}')
#         plt.imshow(outputs[:,:,i])
#     plt.show()
#     plt.clf()
#     plt.close()

# #画像
# n = -1
# # 1層目 
# plot_conv_outputs(conv_outputs[0][n])
# # 2層目 
# plot_conv_outputs(conv_outputs[1][n])
# # 3層目 
# plot_conv_outputs(conv_outputs[2][n])
# # 4層目 
# plot_conv_outputs(conv_outputs[3][n])
# # 5層目 
# plot_conv_outputs(conv_outputs[4][n])
# # 17層目 
# plot_conv_outputs(conv_outputs[16][n])

###正答率のグラフ化 画像とグラフ
model = net
predictions = model.predict(test_data)
judge = [0,0,0,0,0,0,0]
for i in range(len(test_data)):
    #argmaxで二次元配列の列ごとの最大値を示すインデックスを返す
    #予測した値と実際の解
    
    if (judge[int(np.argmax(test_label[i]))] != 0):
        continue
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    bar_label = [0,1,2,3,4,5,6]
    axs[0].imshow(test_data[i],"gray")
    axs[0].set_title(np.argmax(test_label[i]))
    axs[1].bar(bar_label,predictions[i],color="orange",alpha = 0.7)
    axs[1].grid()
    judge[int(np.argmax(test_label[i]))] += 1
    #print(predictions[i],test_label[i])
    plt.show()
    plt.clf()
    plt.close()

#モデルの可視化
plot_model(
    model,
    show_shapes=True,
    show_layer_names=True,
    to_file="model_kfold.png"
)