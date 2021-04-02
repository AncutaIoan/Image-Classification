!unzip ai-unibuc-24-22-2021.zip

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


train = pd.read_csv (r'/content/train.txt', header = None)
train.columns = ['idImg','class']


train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img('train/'+train['idImg'][i], target_size=(32,32,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)


y=train['class'].values
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(32,32,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(keras.layers.normalization.BatchNormalization())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(9, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))



validation=pd.read_csv(r'/content/validation.txt', header = None)
validation.columns = ['idImg','class']
validation_image = []
for i in tqdm(range(validation.shape[0])):
    img = image.load_img('validation/'+validation['idImg'][i], target_size=(32,32,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    validation_image.append(img)
valid = np.array(validation_image)

predictionvalid = model.predict_classes(valid)
print(predictionvalid)
print("\n")
print(validation['class'])


countRight=0.000
countTotal=0.000
for i in tqdm(range(validation.shape[0])):
    if int(validation['class'][i])==predictionvalid[i]:
        #print("\n"+validation['class'][i])
        countRight+=1
    countTotal+=1
print("\n")
print(countRight)
print(countTotal)
print((countRight*100)/countTotal,"%")


test = pd.read_csv (r'/content/test.txt', header = None)
test.columns = ['idImg']

test_image = []
for i in tqdm(range(test.shape[0])):
    img = image.load_img('test/'+test['idImg'][i], target_size=(32,32,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
test_p = np.array(test_image)

#prediction = model.predict_classes(test)
prediction = np.argmax(model.predict(test_p), axis=-1)

for i in range(len(prediction)):
  print(f"{test['idImg'][i]},{prediction[i]}")
