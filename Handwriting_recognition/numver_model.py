import os
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#模型加载
class minist_model(tf.keras.models.Model):
    def __init__(self):
        super(minist_model, self).__init__()
        self.f=tf.keras.layers.Flatten()
        self.d1=tf.keras.layers.Dense(128,activation="relu",kernel_regularizer="l2")
        self.d2=tf.keras.layers.Dense(10,activation='softmax')


    def call(self,input):
        y=self.f(input)
        y=self.d1(y)
        y=self.d2(y)
        return y

checkpoint_path="./checkpoint/"

if not os.path.join(checkpoint_path):
    os.mkdir(checkpoint_path)



#数据集加载
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train/255.
x_test=x_test/255.
c_p=[
    tf.keras.callbacks.EarlyStopping(patience=6,min_delta=1e-2),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path+"mode.ckpt",
                                       save_best_only=True,
                                       save_weights_only=True)
]
#创建模型
model=minist_model()

#模型配置
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)

####将模型进行保存
path="./h5/"
name="mnist.h5"
if not os.path.exists(path):
    os.mkdir(path)
history=model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test),
                  callbacks=c_p)
model.save(path)



plt.figure()
plt.plot(history.history['sparse_categorical_accuracy'],color='blue')
plt.plot(history.history['val_sparse_categorical_accuracy'],color='green')
# plt.plot(history.history['loss'],color='red')
plt.legend(['train_acc','test_acc'])
plt.show()

plt.figure()
plt.plot(history.history['loss'],color='red')
plt.plot(history.history['val_loss'],color="blue")
plt.legend(['loss',"val_loss"])
plt.show()