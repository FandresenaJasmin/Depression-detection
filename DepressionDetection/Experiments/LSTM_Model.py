#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:27:26 2019

@author: emna
"""
import numpy as np
from pathlib import Path
import csv, re, os
import matplotlib.pyplot as plt
import pandas as pd
# Pre-Processing
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# Keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils.np_utils import to_categorical 
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, History
from keras import regularizers

import keras.backend as K
# Tensorflow
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    
def Load_Data():
    # Get Input and Target Data
    input_data = np.ones([1, 57])
    target_data_binary_PHQ = [0]
    target_data_score_PHQ = [0]
    
    i = 1
    target_data_csv = Path('D:/DAIC WOZ # EMNA/full_dataset.csv')
    with open(target_data_csv, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            try:
                participant_ID = str(re.split('\t', row[0])[0])
                binary_PHQ8 = int(re.split('\t', row[1])[0])
                score_PHQ8 = int(re.split('\t', row[2])[0])
                print(" ## \n Participant ID is:{}, with Binary PHQ:{} , and Score PHQ:{} : ", participant_ID, binary_PHQ8, score_PHQ8)
                
                csv_file_path ='D:/DAIC WOZ # EMNA/'+participant_ID+'_P/split/Participant/'
                if os.path.exists(csv_file_path):
                    input_directory = Path(csv_file_path)
                    for my_csv_filename in input_directory.glob("*_AUDIO_*.csv"):  
                        # Extract the .csv filename   
                        my_csv_filename = my_csv_filename.stem
                        print("## \n Retrieving coef matrix from file: ", my_csv_filename)

                        io = pd.read_csv(csv_file_path+my_csv_filename+'.csv', sep=",", usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,60))

                        # Get the Matrix
                        i = i+1
                        io = np.array(io, dtype=np.float32)
                        print("###################################################")
                        
                        input_data = np.append(input_data, io, axis=0)                       
                        for j in range(io.shape[0]):
                            target_data_binary_PHQ.append(binary_PHQ8)
                            target_data_score_PHQ.append(score_PHQ8)
                            
            except ValueError:
                print("Skipping the following line: ", row[0])
    csvFile.close()            
                
    print("the size of my list-input-matrix: ", i)
    
    input_data_final = np.delete(input_data, 0, 0)
    target_data_binary_PHQ_final = np.delete(target_data_binary_PHQ, 0, 0)
    target_data_score_PHQ_final = np.delete(target_data_score_PHQ, 0, 0)
    
    return input_data_final, target_data_binary_PHQ_final, target_data_score_PHQ_final

# Loading data
input_data, target_data_binary_PHQ, target_data_score_PHQ = Load_Data()

# Splitting into Training and Testing datasets
input_data = np.asarray(input_data, dtype=np.float32)
target_bin = np.asarray(target_data_binary_PHQ, dtype=int)
target_score = np.asarray(target_data_score_PHQ, dtype=np.int64)

#"Testing Binary Score"
x_train, x_test, y_train, y_test = train_test_split(input_data, target_bin, test_size = 0.2 , random_state = 42)

#"Testing PHQ scores"
#x_train, x_test, y_train, y_test = train_test_split(input_data, target_score, test_size = 0.2 , random_state = 42)

print("###split complete###")

scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print("Done Scaling!")

# Dimensions Update
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("Done!")


# new LSTM model   
    
reg = regularizers.l1_l2(l1=0.001, l2=0.00)
checkpoints_path = "C:/Users/yobitrust/Desktop/output/weights8.{epoch:02d}-{val_loss:.2f}.hdf5"           

model = Sequential()

# First Layer
model.add(LSTM(units=40, activation='tanh', recurrent_activation='hard_sigmoid', recurrent_dropout=0.002, kernel_initializer='glorot_uniform', input_shape=(x_train.shape[1], 1), return_sequences=True, bias_regularizer=reg))
model.add(BatchNormalization())
Dropout(0.002)

# Second Layer
model.add(LSTM(units=30, activation='tanh',recurrent_activation='hard_sigmoid', recurrent_dropout=0.002, kernel_initializer='glorot_uniform', bias_regularizer=reg, return_sequences=True))
model.add(BatchNormalization())
Dropout(0.002)

"""
# Forth Layer
model.add(LSTM(units=20, activation='tanh', recurrent_activation='sigmoid', kernel_initializer='glorot_uniform', bias_regularizer=reg, return_sequences=True))  
model.add(BatchNormalization())
Dropout(0.02)
"""
# Fifth Layer
model.add(LSTM(units=20, activation='tanh', recurrent_activation='sigmoid', kernel_initializer='glorot_uniform', bias_regularizer=reg))  
model.add(BatchNormalization())
Dropout(0.002)

model.add(Dense(15, activation='tanh'))
model.add(Dense(10, activation='tanh'))

# Sixth Layer
model.add(Dense(1, activation='sigmoid'))

# Compiler
#opt = SGD(lr=0.0001, decay=1e-6, momentum=0.9)
opt = Adam(lr=1e-3, decay=1e-6)
model.compile(optimizer = opt, loss = root_mean_squared_error, metrics=['accuracy'])  

model.summary()

#checkpoint of the model (based on val_loss)
checkpoint = ModelCheckpoint(checkpoints_path, verbose=1, save_weights_only=True, save_best_only=True)

#early stopping
earlyStopping = EarlyStopping(patience = 10, verbose=1)

#learning rate reduction
reducelr = ReduceLROnPlateau(verbose = 1, patience= 4)

#terminating on NaN loss values
nanStopping = TerminateOnNaN()

#saving callback history
savingHistory = History()

#callbacks
callbacks_list = [checkpoint, earlyStopping,reducelr, nanStopping, savingHistory]

history = model.fit(x_train, y_train, epochs = 300, batch_size = 170, validation_data=(x_test, y_test), verbose=1, shuffle=True, callbacks=callbacks_list)


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
