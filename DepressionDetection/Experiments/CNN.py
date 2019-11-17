# -*- coding: utf-8 -*-
"""
@author: Zaineb Abdelli

"""

import numpy as np
from pathlib import Path
import csv, re, os 
import matplotlib.pyplot as plt
from google.colab import drive
import pandas as pd
import math

# Pre-Processing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# Keras
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical 
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error 

from sklearn.utils import shuffle


from scipy.signal import resample

import matplotlib.pyplot as plt

np.random.seed(42)

from sklearn.preprocessing import OneHotEncoder 

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Softmax, Add, Flatten, Activation , Dropout
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
import keras 
from keras import optimizers

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from google.colab import drive
drive.mount('/gdrive')
cd /gdrive

cd /gdrive

def Load_Data():
    # Get Input and Target Data
    input_data = np.empty([1, 60])
    target_data_binary_PHQ = [0]
    target_data_score_PHQ = [0]
    
    i = 1
    target_data_csv = Path('/gdrive/My Drive/Zeineb/half_dataset.csv')
    with open(target_data_csv, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            try:
                participant_ID = str(re.split('\t', row[0])[0])
                binary_PHQ8 = int(re.split('\t', row[1])[0])
                score_PHQ8 = int(re.split('\t', row[2])[0])
                print(" ## \n Participant ID is:{}, with Binary PHQ:{} , and Score PHQ:{} : ", participant_ID, binary_PHQ8, score_PHQ8)
                
                csv_file_path = '/gdrive/My Drive/Zeineb/'+participant_ID+'_P/split/Participant/'   
                if os.path.exists(csv_file_path):
                    input_directory = Path(csv_file_path)
                    for my_csv_filename in input_directory.glob("*_AUDIO_*.csv"):  
                        # Extract the .csv filename   
                        my_csv_filename = my_csv_filename.stem
                        print("## \n Retrieving coef matrix from file: ", my_csv_filename)

                        io = pd.read_csv(csv_file_path+my_csv_filename+'.csv', sep=",", usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60))

                        # Get the Matrix
                        i = i+1
                        io = np.array(io, dtype=np.float64)
                        io -= np.mean(io, axis=0)
                        io /= np.nan_to_num(np.std(io, axis=0), copy=False)
                        print("###################################################")
                        
                        input_data = np.append(input_data, io, axis=0)                       
                        for i in range(io.shape[0]):
                            target_data_binary_PHQ.append(binary_PHQ8)
                            target_data_score_PHQ.append(score_PHQ8)
                            
            except ValueError:
                print("Skipping the following line: ", row[0])
    csvFile.close()            
                
    print("the size of my list-input-matrix: ", i)
    
    return input_data, target_data_binary_PHQ, target_data_score_PHQ

# Loading data
input_data, target_data_binary_PHQ, target_data_score_PHQ = Load_Data()

# Splitting into Training and Testing datasets
np.nan_to_num(input_data, copy=False)
input_data = np.asarray(input_data, dtype=np.float64)
target = np.asarray(target_data_binary_PHQ, dtype=float)
x_train, x_test, y_train, y_test = train_test_split(input_data, target, test_size = 0.2 , random_state = 40)

# Normalizing Training data
np.nan_to_num(x_train, copy=False)
#print(x_train)
x_train -= np.mean(x_train, axis=0)
np.nan_to_num(x_train, copy=False)
#print("####")
#print(x_train)
x_train /= np.std(x_train, axis=0)
np.nan_to_num(x_train, copy=False)
#print("####")
#print(x_train)

# Normalizing Testing data
np.nan_to_num(x_test, copy=False)
#print(x_test)
x_test -= np.mean(x_test, axis=0)
np.nan_to_num(x_test, copy=False)
#print("####")
#print(x_test)
x_test /= np.std(x_test, axis=0)
np.nan_to_num(x_test, copy=False)
#print("####")
#print(x_test)

"""*DATA* VISUALIZATION

PCA DECOMPOSITION

from sklearn.decomposition import PCA

pca = PCA(n_components=100) #close to nbre of feature
pca.fit(x_train)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')

== the plot is useful to determine the best nbre of component that describe data changes
"""

NCOMPONENTS = 30
from sklearn.decomposition import PCA

pca = PCA(n_components=NCOMPONENTS)
X_pca_train = pca.fit_transform(x_train)
X_pca_test = pca.transform(x_test)
pca_std = np.std(X_pca_train)

print(x_train.shape)
print(X_pca_train.shape)

#X_pca_train = np.expand_dims(X_pca_train, -1)
#X_pca_test = np.expand_dims(X_pca_test, -1)
X_train = np.expand_dims(x_train, -1)
X_test = np.expand_dims(x_test, -1)

X_train.shape

"""from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
inv_pca = pca.inverse_transform(X_pca_train)
inv_sc = scaler.inverse_transform(inv_pca)

MODEL
"""

epochs = 70
batch_size = 100

verbose = 1
num_classes = 2

y_train.shape

ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train.reshape(-1,1)) # contient des binaire / 
y_test = ohe.transform(y_test.reshape(-1,1))
print(y_train)

y_train.shape
X_train.shape

ls

from keras.optimizers import RMSprop
from keras.callbacks import TerminateOnNaN, History
from keras.layers.normalization import BatchNormalization

checkpoints_path = "/gdrive/My Drive/check/poids.{epoch:02d}-{val_loss:.2f}.hdf5"

#THE MODEL

n_obs, feature, depth = X_train.shape
batch_size = 200 # refers to the number of training examples utilized in one iteration.
K.clear_session()
inp = Input(shape=(feature, depth)) #used to instantiate a Keras tensor.
C = Conv1D(filters=88, kernel_size=3, strides=1)(inp)

C11 = Conv1D(filters=88, kernel_size=3, strides=1, padding='same')(C)
A11 = Activation("relu")(C11)
C12 = Conv1D(filters=88, kernel_size=3, strides=1, padding='same')(A11)
S11 = Add()([C12, C])
A12 = Activation("relu")(S11)
M11 = MaxPooling1D(pool_size=3, strides=2)(A12)
#BatchNormalization()
Dropout(0.2)

C21 = Conv1D(filters=88, kernel_size=3, strides=1, padding='same')(M11)
A21 = Activation("relu")(C21)
C22 = Conv1D(filters=88, kernel_size=3, strides=1, padding='same')(A21)
S21 = Add()([C22, M11])
A22 = Activation("relu")(S11)
M21 = MaxPooling1D(pool_size=3, strides=2)(A22)
Dropout(0.2)

C31 = Conv1D(filters=88, kernel_size=3, strides=1, padding='same')(M21)
A31 = Activation("relu")(C31)
C32 = Conv1D(filters=88, kernel_size=3, strides=1, padding='same')(A31)
S31 = Add()([C32, M21])
A32 = Activation("relu")(S31)
M31 = MaxPooling1D(pool_size=3, strides=2)(A32)
#BatchNormalization()
Dropout(0.2)
""""
C41 = Conv1D(filters=32, kernel_size=5, strides=2, padding='same')(M31)
A41 = Activation("relu")(C41)
C42 = Conv1D(filters=32, kernel_size=5, strides=2, padding='same')(A41)
S41 = Add()([C42, M31])
A42 = Activation("relu")(S41)
M41 = MaxPooling1D(pool_size=5, strides=2)(A42)
Dropout(0.2)

C51 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M41)
A51 = Activation("relu")(C51)
C52 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A51)
S51 = Add()([C52, M41])
A52 = Activation("relu")(S51)
M51 = MaxPooling1D(pool_size=5, strides=2)(A52)
Dropout(0.2)
"""
F1 = Flatten()(M31)

D1 = Dense(2)(F1)
A6 = Activation("relu")(D1)
D2 = Dense(32)(A6)
D3 = Dense(2)(D2)
A7 = Softmax()(D1)

model = Model(inputs=inp, outputs=A7)

model.summary()
X_train.shape

def exp_decay(epoch): #One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
    initial_lrate = 0.0001
    k = 0.75
    t = n_obs//(10000 * batch_size)  # every epoch we do n_obs/batch_size iteration
    lrate = initial_lrate * math.exp(-k*t) #we reduce the learning rate by a constant factor every few epochs.
    return lrate
  
#Iterations is the number of batches needed to complete one epoch.
lrate = LearningRateScheduler(exp_decay) #a callback that define a function to invoke during the execution

adam = Adam(lr = 0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) #Before training a model, you need to configure the learning process

#checkpoint of the model (based on val_loss)
checkpoint = ModelCheckpoint(checkpoints_path, verbose=1, save_weights_only=True, save_best_only=True)

#early stopping
earlyStopping = EarlyStopping(patience = 10, verbose=1)

#terminating on NaN loss values
nanStopping = TerminateOnNaN()

#saving callback history
savingHistory = History()

#callbacks
callbacks_list = [checkpoint, earlyStopping,lrate, nanStopping, savingHistory]
import time 
start_time = time.time()
history = model.fit(X_train, y_train, 
                    epochs=75, 
                    batch_size=batch_size, 
                    verbose=2, 
                    validation_data=(X_test, y_test),
                    shuffle=True,
                    callbacks=callbacks_list)                    

print("training time {}".format(time.time()-start_time))

y_pred = model.predict(X_test, batch_size=200)

print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# Network Performances Display
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
 
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()

