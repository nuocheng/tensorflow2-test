import tensorflow as tf
import numpy as np
import cv2
from PIL import Image,ImageFile


def change_image_size():
    #图片压缩为某一个大小
    infile="./image/4.jpg"
    outfile="./image/4-1.jpg"

    ImageFile.LOAD_TRUNCATED_IMAGES = True  # 防止图像被截断而报错
    im=Image.open(infile)
    out=im.resize((28,28),Image.ANTIALIAS)
    out.save(outfile)
# change_image_size()
model=tf.keras.models.load_model("./h5")
data=cv2.imread("./image/3-1.jpg")
data = cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)
cv2.imwrite("./image/3-2.jpg", data, [int(cv2.IMWRITE_JPEG_QUALITY),95])

data=tf.cast(data,tf.float32)
data=data/255.
data=tf.expand_dims(data,0)


print(data.shape)
index=[0,1,2,3,4,5,6,7,8,9]
print(model.predict(data))
print(tf.argmax(model.predict(data),axis=1))
print(index[tf.argmax(model.predict(data),axis=1).numpy()[0]])