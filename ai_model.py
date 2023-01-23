import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score
import glob
import os

# Create a TensorBoard callback

def process_data(file_name):

    logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    # Load stock data
    data = pd.read_csv(file_name)

    # Scale the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.values)

    # Split the data into training and testing sets
    train_data = data[:int(data.shape[0]*0.8), :]
    test_data = data[int(data.shape[0]*0.8):, :]

    # get the x_train and y_train
    x_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    # reshape the data for LSTM
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], 1, 1))

    # Build the LSTM model
    model = Sequential()
    # get the list of all h5 files

    model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1],1), 
                kernel_initializer='glorot_normal', recurrent_initializer='orthogonal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(LSTM(units=100, return_sequences=True,
                kernel_initializer='glorot_normal', recurrent_initializer='orthogonal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(LSTM(units=100,
                kernel_initializer='glorot_normal', recurrent_initializer='orthogonal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(8))

    h5_files = glob.glob("best_model*.h5")
    # get the latest file
    latest_file = max(h5_files, key=os.path.getctime)
    
    # load the latest weights
    model.load_weights(latest_file)

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_mae', patience=10)
    # model_checkpoint = ModelCheckpoint(filepath='best_model.h5', save_best_only=True, save_weights_only=True)
    model_checkpoint = ModelCheckpoint(filepath='best_model_{}.h5'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), save_best_only=True, save_weights_only=True)

    # Fit the model to the training data
    model.fit(x_train, y_train, epochs=100, batch_size=128,
            validation_data=(x_test, y_test), callbacks=[early_stopping, model_checkpoint, tensorboard_callback])

    # Make predictions on the test data
    test_predictions = model.predict(x_test, steps=300)

    # Unscale the predictions

    test_predictions = test_predictions.reshape(-1,8)
    # test_predictions = scaler.inverse_transform(test_predictions)
    # test_predictions = test_predictions.reshape(-1,1)
    test_predictions = scaler.inverse_transform(test_predictions)

    f= open(f"predictions/output_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt","w+")
    dates= []
    for prediction in test_predictions:
        # Convert timestamp to datetime
        timestamp = prediction[6]
        timestamp = datetime.fromtimestamp(timestamp / 1e3)

        # Add the timedelta
        date_time_string = timestamp.strftime("%Y-%m-%d")
        
        f.write(f"{date_time_string}\n")
        f.write(f"\tPrice Open: ${int(prediction[2])}\n")
        f.write(f"\tPrice Close: ${int(prediction[3])}\n")
        f.write(f"\tPrice High: ${int(prediction[4])}\n")
        f.write(f"\tPrice Low: ${int(prediction[5])}\n")
        f.write("--------------------\n")

    print("=============================")
    print("PREDICTIONS COMPLETE")
    print("=============================")
    print(model.summary())