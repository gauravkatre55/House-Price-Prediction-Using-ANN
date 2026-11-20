import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load the dataset
data = pd.read_csv('data/california_housing.csv')

# create features and target variable

x=data.drop('MedHouseVal', axis=1)
y=data['MedHouseVal']

# create training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Standardize the features
scaler = StandardScaler()
x_train_sc=scaler.fit_transform(x_train)
x_test_sc=scaler.transform(x_test)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
## ANN Implementation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from tensorflow.keras.optimizers import Adam
import datetime

# Build the model
model=Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_train_sc.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()

# create a optimizer
import tensorflow.keras.losses
opt=Adam(learning_rate=0.001)
loss=tensorflow.keras.losses.MeanSquaredError()

# compile the model
model.compile(optimizer=opt, loss=loss, metrics=['mae'])

# setup early Tensboard 
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
tensorboard = TensorBoard(log_dir="logs/{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))) 

#setup early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

### Train the model
history=model.fit(x_train_sc, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping, tensorboard])

# Predictions 
y_pred = model.predict(x_test_sc).flatten() 

# Regression evaluation metrics 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred) 
mse = mean_squared_error(y_test, y_pred) 
rmse = np.sqrt(mse) 
r2 = r2_score(y_test, y_pred) 
print("MAE:", mae) 
print("MSE:", mse) 
print("RMSE:", rmse) 
print("R2 Score:", r2)

# save the model
model.save('house_price_model.h5')