# -*- coding: utf-8 -*-
"""Copy of Copy of my_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ukAsidD1adIGYneFXwf0KU-wE3Ljxnuc

# Apparent Age and Gender Prediction in Keras


https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/
"""

import os
import scipy.io
import pickle
import numpy as np
import pandas as pd
#from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# memory footprint support libraries/code
!ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
!pip install gputil
!pip install psutil
!pip install humanize
import psutil
import humanize
import os
import GPUtil as GPU
GPUs = GPU.getGPUs()
# XXX: only one GPU on Colab and isnt guaranteed
#gpu = GPUs[0]
#def printm():
#    process = psutil.Process(os.getpid())
#    print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
#    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(GPUs.memoryFree, GPUs.memoryUsed, GPUs.memoryUtil*100, GPUs.memoryTotal))
#printm()

GPUs

from google.colab import drive
drive.mount('/content/drive')

"""#### Очистка данных
Некоторые изображения не включают людей в набор данных вики. Например, изображение вазы существует в наборе данных. Более того, на некоторых фотографиях может быть два человека. Кроме того, некоторые взяты на расстоянии. Номинальная стоимость может помочь нам понять, ясна картинка или нет. Кроме того, информация о возрасте отсутствует для некоторых записей. Все они могут спутать модель. Мы должны их игнорировать. Наконец, ненужные столбцы должны быть удалены, чтобы занимать меньше памяти

Некоторые фотографии сделаны для нерожденных людей. Возрастное значение представляется отрицательным для некоторых записей. Грязные данные могут быть причиной этого. Более того, некоторые, кажется, живут более 100 лет. Мы должны ограничить проблему прогнозирования возраста от 0 до 100 лет.
"""

import pandas as pd
df =  pd.read_csv("/content/drive/My Drive/Colab Notebooks/vgg/geekhub-2020-age-estimation-challenge/train.csv")
df

y = (df['age']).values
histogram_age = df['age'].hist(bins=df['age'].nunique(), figsize=(15, 15))

namb =['0000%s' % i for i in range(0, 10)] + ['000%s' % j for j in range(10, 100)] + ['00%s' % j for j in range(100, 686)]
  
df['namb']  = namb
print(df)

# Commented out IPython magic to ensure Python compatibility.
# %%time 
# from glob import glob
# #В столбце «Полный путь» указывается точное местоположение изображения на диске. Нам нужны его значения пикселей.
# #tf.keras.preprocessing.image.load_img
# target_size = (224, 224)
# x = [] 
# def getImagePixels(image_path):
#    # for pic in glob("/content/drive/My Drive/Colab Notebooks/vgg/geekhub-2020-age-estimation-challenge/train_images/train_images/train_images/*"):
#     img = image.load_img("/content/drive/My Drive/Colab Notebooks/vgg/geekhub-2020-age-estimation-challenge/train_images/%s.jpg" % image_path,
#                          grayscale=False, target_size=target_size)
#     #print(img)
#     x = image.img_to_array(img).reshape(1, -1)[0]
#     #print(img)
#     return x
# 
# 
# df['pixels'] = df['namb'].apply(getImagePixels)
# 
# #Мы можем извлечь реальные значения пикселей изображения
#

df.head()

df

"""#### Модель прогнозирования видимого возраста
Прогноз возраста - проблема регрессии. Но исследователи определяют это как проблему классификации. В выходном слое 101 класс для возрастов от 0 до 100. Они применили трансферное обучение. для этой обязанности. Их выбор был VGG для imagenet.
#### Подготовка ввода вывода
Фрейм данных Pandas включает в себя как входную, так и выходную информацию для задач прогнозирования возраста и пола. Ви должны просто сосредоточиться на возрастной задаче.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# classes = 101 #0 to 100
# #target =np.arange(101)
# target = df['age'].values
# target_classes = tf.keras.utils.to_categorical(target, classes)
#

features = []
 
for i in range(0, df.shape[0]):
    features.append(df['pixels'].values[i])

features = np.array(features)
features = features.reshape(features.shape[0], 224, 224, 3)

features.shape

target_classes.shape

len(range(df.shape[0]))

# Commented out IPython magic to ensure Python compatibility.
# %%time
# #Также нам нужно разделить набор данных на обучающий и тестовый набор.
# from sklearn.model_selection import train_test_split
# train_x, test_x, train_y, test_y = train_test_split(features, target_classes, test_size=0.30)

"""Окончательный набор данных состоит из 22578 экземпляров. Он разделен на 15905 экземпляров поездов и 6673 экземпляров тестов.
#### Передача обучения
Как уже упоминалось, исследователь использовал модель VGG imagenet. Тем не менее, они настроили вес для этого набора данных. При этом, я предпочитаю использовать VGG-Face модель . Потому что эта модель настроена на задачу распознавания лиц. Таким образом, мы можем получить результаты для моделей человеческого лица.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# #Face model
# 
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import  ZeroPadding2D, Convolution2D, MaxPooling2D
# from tensorflow.keras import Model
# from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, ZeroPadding2D
# 
# model = Sequential()
# model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
# model.add(Convolution2D(64, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))
#  
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(128, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))
#  
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(256, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(256, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(256, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))
#  
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))
#  
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))
#  
# model.add(Convolution2D(4096, (7, 7), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Convolution2D(4096, (1, 1), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Convolution2D(2622, (1, 1)))
# model.add(Flatten())
# model.add(Activation('softmax'))
# 
#



# Commented out IPython magic to ensure Python compatibility.
# %%time
# #Загрузите предварительно обученные веса для модели VGG-Face. Вы можете найти соответствующую запись в блоге здесь .
# #pre-trained weights of vgg-face model.
# #you can find it here: https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
# #related blog post: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
# model.load_weights('/content/drive/My Drive/Colab Notebooks/vgg/vgg_face_weights.h5')

"""Мы должны зафиксировать веса слоев для ранних слоев, потому что они уже могут обнаружить некоторые шаблоны. Установка сети с нуля может привести к потере этой важной информации. Я предпочитаю заморозить все слои, кроме последних 3 слоев свертки (за исключением последних 7 модулей model.add). Кроме того, я сократил последний слой свертки, потому что у него есть 2622 единицы. Мне нужно всего 101 (возраст от 0 до 100) единиц для задачи прогнозирования возраста. Затем добавьте пользовательский слой свертки, состоящий из 101 единицы."""

for layer in model.layers[:-7]:
    layer.trainable = False
 
base_model_output = Sequential()
base_model_output = Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
base_model_output = Flatten()(base_model_output)
base_model_output = Activation('softmax')(base_model_output)
 
age_model = Model(inputs=model.input, outputs=base_model_output)

"""#### Обучение
Это проблема классификации нескольких классов. Функция потери должна быть категорической кроссентропией . Алгоритм оптимизации будет Адам, чтобы сходить потери быстрее. Я создаю контрольную точку, чтобы контролир
овать модель на протяжении итераций и избегать переоснащения. Итерация с минимальным значением потерь при проверке будет включать в себя оптимальные веса. Поэтому я буду следить за потерями при проверке и сохраню только самую лучшую.

Чтобы избежать переобучения, я кормлю случайные 256 экземпляров для каждой эпохи.
"""

#check trainable layers
if False:
    for layer in model.layers:
        print(layer, layer.trainable)
    
    print("------------------------")
    for layer in age_model.layers:
        print(layer, layer.trainable)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# #sgd = keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
# age_model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['mae'])
#

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='age_model.hdf5',
                               monitor = "val_loss", verbose=1, save_best_only=True, mode = 'auto')

scores = []

enableFit = False

if enableFit:
    epochs = 250
    batch_size = 256
    
    for i in range(epochs):
        print("epoch ",i)
 
        ix_train = np.random.choice(train_x.shape[0], size=batch_size)
 
        score = age_model.fit(train_x[ix_train], 
                          train_y[ix_train], 
                          epochs=1, 
                          validation_data=(test_x, test_y), 
                          callbacks=[checkpointer])
 
        scores.append(score)
    
    #restore the best weights
    from keras.models import load_model
    age_model = load_model("classification_age_model.hdf5")
    
    age_model.save_weights('/content/drive/My Drive/Colab Notebooks/vgg/age_model_weights.h5')
        
else:
    #pre-trained weights for age prediction: https://drive.google.com/file/d/1YCox_4kJ-BYeXq27uUbasu--yz28zUMV/view?usp=sharing
    age_model.load_weights("/content/drive/My Drive/Colab Notebooks/vgg/age_model_weights.h5")

"""Кажется, что потери проверки достигают минимума. Увеличение эпох приведет к переоснащению.

#### Оценка модели на тестовом наборе
Мы можем оценить окончательную модель на тестовом наборе.
Это дает как потерю достоверности, так и точность соответственно. Похоже, у нас есть следующие результаты.
"""

#loss and accuracy on validation set
age_model.evaluate(test_x, test_y, verbose=1)

predictions = age_model.predict(test_x)

output_indexes = np.array([i for i in range(0, 101)])
apparent_predictions = np.sum(predictions * output_indexes, axis = 1)

mae = 0
actual_mean=[]
for i in range(0 ,apparent_predictions.shape[0]):
    prediction = int(apparent_predictions[i])
    actual = np.argmax(test_y[i])
    
    abs_error = abs(prediction - actual)
    actual_mean = actual_mean + actual
    
    mae = mae + abs_error
    
mae = mae / apparent_predictions.shape[0]

print("mae: ",mae)
print("instances: ",apparent_predictions.shape[0])

scores

# Testing model on a custom image

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from glob import glob
picture=glob('/content/drive/My Drive/Colab Notebooks/vgg/geekhub-2020-age-estimation-challenge/test_images/*')
picture

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
 
def loadImage(filepath):
  test_img = image.load_img(filepath, target_size=(224, 224))
  test_img = image.img_to_array(test_img)
  test_img = np.expand_dims(test_img, axis = 0)
  test_img /= 255
  return test_img
 
picture = "/content/drive/My Drive/Colab Notebooks/vgg/geekhub-2020-age-estimation-challenge/test_images/00987.jpg"
prediction = age_model.predict(loadImage(picture))

import cv2
from tf.keras.model.metrics_names import predict_classes
pred=[]
for img in glob("/content/drive/My Drive/Colab Notebooks/vgg/geekhub-2020-age-estimation-challenge/test_images/*.jpg") :
    age_model.predict_classes(icv2.imread(img)))
    pred.append(img)
pred

import cv2
p=[]
[p.append(cv2.imread(x)) for x in
  glob('/content/drive/My Drive/Colab Notebooks/vgg/geekhub-2020-age-estimation-challenge/test_images/*')]

y_proba = model.predict(x)
y_classes = tf.keras.np_utils.probas_to_classes(y_proba)

y_prob = model.predict(x) 
y_classes = y_prob.argmax(axis=-1)

y_pos = np.arange(101)
plt.bar(y_pos, prediction[0], align='center', alpha=0.3)
plt.ylabel('percentage')
plt.title('age')
plt.show()

pd.Series( age_model.predict_proba(loadImage(picture))[:, 1], 
          name='d_15').to_csv('logit_2feat.csv', 
                                           index_label='id', 
                                           header=True)

target_size = (224, 224)
x_test =[] 
for pic in glob("/content/drive/My Drive/Colab Notebooks/vgg/geekhub-2020-age-estimation-challenge/test_images/*.jpg") :
    img = image.load_img(pic, grayscale=False, target_size=target_size)
    print(img)
    x_test.append(img)
    
    
x_test

import glob
import cv2

for img in glob.glob("/content/drive/My Drive/Colab Notebooks/vgg/geekhub-2020-age-estimation-challenge/test_images/*.jpg"):
    #test_x.append(np.asarray(Image.open(img)[:, :, :3]))
    #test_x= cv2.imread(img)
    test_x.append(Image.open(img))

np.stack(test_x).astype(np.float32)

import glob
prediction=[]
p=[]
for i in  glob.glob("/content/drive/My Drive/Colab Notebooks/vgg/geekhub-2020-age-estimation-challenge/test_images/*"):
    prediction = age_model.predict(loadImage(i))
    p.append(prediction)

p

