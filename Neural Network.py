import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
from sklearn.model_selection import train_test_split
import tensorflow 
from tensorflow import keras 
from keras import Sequential
from keras.layers import Dense,Dropout
import keras_tuner as kt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


df=pd.read_csv('diabetes.csv')

#Building A Model
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

x=scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

model=Sequential()
model.add(Dense(32,activation='relu',input_dim=8))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
#model.fit(x_train,y_train,batch_size=32,epochs=100,validation_data=(x_test,y_test))

def build_model(hp):
    model=Sequential()
    model.add(Dense(32,activation='relu',input_dim=8))
    model.add(Dense(1,activation='sigmoid'))
    optimizer=hp.Choice('optimizer',values=['adam','sgd','rmsprop','adadelta'])
    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return model

tuner=kt.RandomSearch(build_model,
                      objective='val_accuracy',
                      max_trials=5)

tuner.search(x_train,y_train,epochs=5,validation_data=(x_test,y_test))

tuner.get_best_hyperparameters()[0].values

model=tuner.get_best_models(num_models=1)[0]
model.summary()
model.fit(x_train,y_train,batch_size=32,epochs=100,initial_epoch=100,validation_data=(x_test,y_test))

def build_model(hp):

    model=Sequential()
    units=hp.Int('units',min_value=8,max_value=128,step=8)
    model.add(Dense(units=units,activation='relu',input_dim=8))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    return model

tuner=kt.RandomSearch(build_model,
                      objective='val_accuracy',
                      max_trials=5,
                      directory='mydir')

tuner.search(x_train,y_train,epochs=5,validation_data=(x_test,y_test))

tuner.get_best_hyperparameters()[0].values

model=tuner.get_best_models(num_models=1)[0]
print(model.fit(x_train,y_train,batch_size=32,epochs=100,initial_epoch=100))

def build_model(hp):
    model=Sequential()
    model.add(Dense(40,activation='relu',input_dim=8))

    for i in range(hp.Int('num_layers',min_value=1,max_value=10)):
        model.add(Dense(40,activation='relu'))
    
    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

    return model

tuner=kt.RandomSearch(build_model,
                      objective='val_accuracy',
                      max_trials=3,
                      directory='num_layers')

tuner.search(x_train,y_train,epochs=5,validation_data=(x_train,y_train))
tuner.get_best_hyperparameters()[0].values
model=tuner.get_best_models(num_models=1)[0]
model.fit(x_train,y_train,epochs=100,initial_epoch=6,validation_data=(x_train,y_train))

def build_model(hp):
    model=Sequential()
    counter=0


    for i in range(hp.Int('num_layers',min_value=1,max_value=10)):
        if counter==0:
            model.add(
                Dense(
                    hp.Int('units'+str(i),min_value=8,max_value=128,step=8),
                    activation=hp.Choice('activation'+str(i),values=['relu','tanh','sigmoid']),
                    input_dim=8
                    )
                )
            model.add(Dropout(hp.Choice('Dropout'+str(i),values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])))
    else:
        model.add(Dense(
                    hp.Int('units'+str(i),min_value=8,max_value=128,step=8),
                    activation=hp.Choice('activation'+str(i),values=['relu','tanh','sigmoid']),
                    )
                )
        model.add(Dropout(hp.Choice('Dropout'+str(i),values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])))
    counter+=1
    
    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer=hp.Choice('optimizer',values=['adam','sgd','rmsprop','adadelta']),loss='binary_crossentropy',metrics=['accuracy'])
    return model

tuner=kt.RandomSearch(build_model,
                      objective='val_accuracy',
                      max_trials=3,
                      directory='mydir',
                      project_name='final1')

        
tuner.search(x_train,y_train,epochs=5,validation_data=(x_train,y_train))

print(tuner.get_best_hyperparameters()[0].values)

model=tuner.get_best_models(num_models=1)[0]
model.fit(x_train,y_train,epochs=200,initial_epoch=6,validation_data=(x_train,y_train))