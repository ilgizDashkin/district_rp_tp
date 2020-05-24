import json
import re
import numpy as np
import pandas as pd
import sklearn 
import matplotlib
from matplotlib import pyplot as plt

# это первая часть в которой мы готовим данные из json файла

# # загружаем не ощищеный json
# df=pd.read_json("adres.json")
# # замена латинской P и T на русские Р и Т и убрать лишние пробелы
# # эти латинские символы из ЦРЭС
# df.name = df.name.str.replace('P', 'Р')
# df.name = df.name.str.replace('T', 'Т')
# df.name = df.name.str.replace(' Р', 'Р')
# df.name = df.name.str.replace(' Т', 'Т')
# df.name = df.name.str.replace('П ', 'П-')

# df.to_excel("adres.xls")
# df.to_json("adres_clear.json")

# # загружаем очищенные данные
# df=pd.read_json("adres_clear.json")
# df.describe()

# # возьмем только районы и имя рп-тп
# district_and_name=df[['district','name']]
# district_and_name.describe()
# district_and_name.to_json("district_and_name.json")
# district_and_name['num_code'] = df.name.str.replace('РП-', '1')
# district_and_name['num_code'] = district_and_name['num_code'].str.replace('ТП-', '2')
# district_and_name['num_code'] = district_and_name['num_code'].str.replace('ПП-', '3')
# # оставить только числа
# district_and_name['num_code'] = district_and_name['num_code'].str.replace(r'[^0-9]', '')
# # дополняем строку нулями справа
# district_and_name['num_code'] =district_and_name['num_code'].str.ljust(5, '0')
# district_and_name['num_district'] =district_and_name['district'].map({'ЦРЭС': '1', 'ЮРЭС': '2','ЗРЭС': '3','СРЭС': '4','ВРЭС': '5','ЮВРЭС': '6'})
# district_and_name[250:300]
# district_and_name.to_excel("district_and_name.xls")
# district_and_name.describe()
# district_and_name

# конец первой части

# здесь начинается нейронка

# def pos_of_str(x):
#   x=str(x)
#   if len(x)!=3:
#     print(len(x))
#     print(x)
#   arr=[]
#   for i in x:
#     arr.append(int(i))
#   return arr

# берем получившийся файл и создаем датасет для обучения
df=pd.read_excel("district_and_name.xls", index_col=0)
# превращаем колонку num_code в одномерные тензоры
# df['num_code']=df['num_code'].apply(pos_of_str)
df.info()
df.dtypes
# раскидываем num_code по колонкам
df['num_code0']=df['num_code'].apply(lambda x: str(x)[0]).astype('int')
df['num_code1']=df['num_code'].apply(lambda x: str(x)[1]).astype('int')
df['num_code2']=df['num_code'].apply(lambda x: str(x)[2]).astype('int')
df['num_code3']=df['num_code'].apply(lambda x: str(x)[3]).astype('int')
df['num_code4']=df['num_code'].apply(lambda x: str(x)[4]).astype('int')

data=df[["name","num_code0","num_code1","num_code2","num_code3","num_code4",'num_district']]
data.info()
X,y=data[["num_code0","num_code1","num_code2","num_code3","num_code4"]].values, np.array(data.num_district)

# разделяем данные на обучающие и проверочные
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(
    X, y, random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train=X
y_train=y

import tensorflow as tf
print('версия тензорфлоу ',tf.__version__)
# from tensorflow.keras import regularizers

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(32, activation='relu',input_shape=(5,)),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(optimizer='rmsprop',
 loss='sparse_categorical_crossentropy', metrics=['acc'])

# history = model.fit(x_train, y_train, epochs=50)
history = model.fit(x_train, y_train,
                    epochs=70,
                    batch_size=20,
                    validation_data=(x_test, y_test))

# # # Дообучим в конце на всем наборе данных
# x_train=X
# y_train=y
# history = model.fit(x_train, y_train,
#                     epochs=20,
#                     batch_size=20,
#                     validation_data=(x_test, y_test))                    

# вывод потерь и точность
model.evaluate(x_test, y_test)
# вывод самой модели
model.summary()

# вывод на график
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='обучающая потеря')
plt.plot(epochs, val_loss, 'b', label='проверочная потеря')
plt.title('обучаючая и проверочная потеря')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='обучающая точность')
plt.plot(epochs, val_acc, 'b', label='проверочная точность')
plt.title('обучающая и проверочная точность')
plt.xlabel('Epochs')
plt.ylabel('точность')
plt.legend()
plt.show()


# # попробуем предсказать ТП-2222
# pred = model.predict([[2,2,2,2,2]])

# # попробуем вывести 100 предсказаний
# pred = model.predict(x_test)
# district=['нулевой','ЦРЭС', 'ЮРЭС','ЗРЭС','СРЭС','ВРЭС','ЮВРЭС']
# for i in range(100):
#   print(i)
#   print(x_test[i])
#   print(pred[i])
#   k=0
#   for j in pred[i]:
#     if j>0.1:
#       print('вероятность ',j*100,'% ', district[k])
#     k+=1

#   print("Предсказанный район:", district[np.argmax(pred[i])], ", правильный район: ",district[y_test[i]])

# # # можно вывести предсказания для тестового набора
# district=['нулевой','ЦРЭС', 'ЮРЭС','ЗРЭС','СРЭС','ВРЭС','ЮВРЭС']
# pred_class=model.predict_classes(x_test)
# index=0
# for pr in pred_class:
#   print('предсказания ', district[pr],'правильные ответы ',district[y_test[index]])
#   index+=1

# # проверяем работает ли тензорфлоу на gpu
# from tensorflow.python.client import device_lib 
# print(device_lib.list_local_devices())
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# # сохранить модель
# def save_model(model=model,name_model='my_model.h5'):
#     model.save(name_model)  # creates a HDF5 file 'my_model.h5'
#     print('model save in '+name_model)

# save_model()

# # загрузить модель
# from tensorflow.keras.models import load_model
# model_1 = load_model('my_model.h5')
# model_1.summary()
# model_1.predict([[2,2,2,2,2]])


# # конвертировать обученную модель в js
# import tensorflowjs as tfjs
# tfjs.converters.save_keras_model(model,'./model/')