import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score # R^2 score
from sklearn.metrics import mean_squared_error # squared = True for MSE, False for RMSE
from sklearn.metrics import mean_absolute_error # mean absolute error
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from collections import deque
import plotly.graph_objects as go

class LTSM:
    def __init__(self, df, n_steps=None, lookup_step=None, scale=None, shuffle=None,                                    split_by_date=None, test_size=None, features=None):
        
        self.df = df
        self.data = {}
        
        if n_steps == None:
            self.n_steps = 30 # Window size or the sequence length.
        else:
            self.n_steps = n_steps

        if lookup_step == None:
            self.lookup_step = 1 # Lookup step, 1 is the next day.
        else:
            self.lookup_step = lookup_step

        if scale == None:
            self.scale = True # whether to scale feature columns & output price as well
        else:
            self.scale = scale
        self.scale_str = f"sc-{int(self.scale)}"
        
        if shuffle == None:
            self.shuffle = True # whether to shuffle the dataset
        else:
            self.shuffle = shuffle
        self.shuffle_str = f"sh-{int(self.shuffle)}"
        
        if split_by_date == None:
            self.split_by_date = False  # whether to split the training/testing set by date
        else:
            self.split_by_date = split_by_date

        self.split_by_date_str = f"sbd-{int(self.split_by_date)}"
        
        if test_size == None:
            self.test_size = 0.2 # test ratio size, 0.2 is 20%
        else:
            self.test_size = test_size

        if features == None:
            self.features = ["open", "high", "low", "close", "macd", "rsi", "adx", "cci"] 
        else:
            self.features = features

        # create these folders if they does not exist
        if not os.path.isdir("results"):
            os.mkdir("results")
        if not os.path.isdir("logs"):
            os.mkdir("logs")
        if not os.path.isdir("data"):
            os.mkdir("data")


    def shuffle_in_unison(self, a, b):
        # shuffle two arrays in the same way
        state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(state)
        np.random.shuffle(b)

    def load_data(self): 
        
        self.data['df'] = self.df.copy(deep=True) # Copy of original dataframe.
        for col in self.features: # Validate features exist in the dataframe.
            assert col in self.df.columns, f"'{col}' does not exist in the dataframe."

        if "date" not in self.df.columns: # Add date column if it doesn't exist.
            self.df["date"] = self.df.index

        if self.scale: # Scale the data (prices) from 0 to 1.
            column_scaler = {}
            for column in self.features:
                scaler = preprocessing.MinMaxScaler()
                self.df[column] = scaler.fit_transform(np.expand_dims(self.df[column].values,                                                                               axis=1))
                column_scaler[column] = scaler 
            self.data["column_scaler"] = column_scaler  # So we can reverse the scaled values later.
        
        self.df['future'] = self.df['close'].shift(-self.lookup_step) # Add the target column.

        # last `lookup_step` columns contains NaN in future column
        # get them before droping NaNs
        last_sequence = np.array(self.df[self.features].tail(self.lookup_step))
        self.df.dropna(inplace=True)
        sequence_data = []
        sequences = deque(maxlen=self.n_steps) # Double ended queue with size of "n_steps" (window lengths).

        for entry,target in zip(self.df[self.features + ["date"]].values, self.df['future'].values):
            sequences.append(entry)
            if len(sequences) == self.n_steps:
                sequence_data.append([np.array(sequences), target])

        # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
        # for instance, if n_steps=30 and lookup_step=1, last_sequence should be of 31 (that is 30+1) length
        # this last_sequence will be used to predict future stock prices that are not available in the dataset
        last_sequence = list([s[:len(self.features)] for s in sequences]) + list(last_sequence)
        last_sequence = np.array(last_sequence).astype(np.float32)

        self.data['last_sequence'] = last_sequence
        
        # construct the X's and y's
        X, y = [], []
        for seq, target in sequence_data:
            X.append(seq)
            y.append(target)

        # convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        if self.split_by_date:
            # split the dataset into training & testing sets by date (not randomly splitting)
            train_samples = int((1 - self.test_size) * len(X))
            self.data["X_train"] = X[:train_samples]
            self.data["y_train"] = y[:train_samples]
            self.data["X_test"]  = X[train_samples:]            
            self.data["y_test"]  = y[train_samples:]
            if self.shuffle:
                # shuffle the datasets for training (if shuffle parameter is set)
                self.shuffle_in_unison(self.data["X_train"], self.data["y_train"])
                self.shuffle_in_unison(self.data["X_test"], self.data["y_test"])
        else:    
            # split the dataset randomly
            self.data["X_train"], self.data["X_test"], self.data["y_train"], self.data["y_test"] = \
                train_test_split(X, y, test_size=self.test_size, shuffle=self.shuffle)

        dates = self.data["X_test"][:, -1, -1] # get the list of test set dates
        self.data["test_df"] = self.data["df"].loc[dates] # retrieve test features from the original dataframe
        self.data["test_df"] = self.data["test_df"][~self.data["test_df"].index.duplicated(keep='first')] # Duplicated dates
        # remove dates from the training/testing sets & convert to float32
        self.data["X_train"] = self.data["X_train"][:, :, :len(self.features)].astype(np.float32) 
        self.data["X_test"] = self.data["X_test"][:, :, :len(self.features)].astype(np.float32)

        self.data['df'].to_csv('./rnn-example.csv')
        # return result

    def create_model(self, units=256, cell=LSTM, n_layers=3, dropout=0.3, loss="huber_loss", optimizer="adam",                                    bidirectional=False):

        ### model parameters
        sequence_length = self.n_steps
        n_features = len(self.features)
        self.model = Sequential()
        for i in range(n_layers):
            if i == 0:
                # first layer
                if bidirectional:
                    self.model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None,                                                   sequence_length, n_features)))
                else:
                    self.model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length,n_features)))
            elif i == n_layers - 1:
                # last layer
                if bidirectional:
                    self.model.add(Bidirectional(cell(units, return_sequences=False)))
                else:
                    self.model.add(cell(units, return_sequences=False))
            else:
                # hidden layers
                if bidirectional:
                    self.model.add(Bidirectional(cell(units, return_sequences=True)))
                else:
                    self.model.add(cell(units, return_sequences=True))
            # add dropout after each layer
            self.model.add(Dropout(dropout))
        self.model.add(Dense(1, activation="linear"))
        self.model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
        # return model

    def train(self, batch_size=64, epochs=500):
        checkpointer = ModelCheckpoint(os.path.join("results", 'rnn-example' + ".h5"), save_weights_only=True,                                                      save_best_only=True, verbose=0) # TF callbacks
        tensorboard = TensorBoard(log_dir=os.path.join("logs", 'rnn-example'))
        self.history = self.model.fit(self.data["X_train"], self.data["y_train"],
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(self.data["X_test"], self.data["y_test"]),
                                        callbacks=[checkpointer, tensorboard], verbose=0)

    def predict(self, X_test = None):
        if X_test == None:        
            model_path = os.path.join('results', 'rnn-example') + '.h5'
            y_hat = self.model.predict(self.data['X_test'])
            
            y_test = np.squeeze(self.data["column_scaler"]["close"].inverse_transform(
                np.expand_dims(self.data['y_test'], axis=0)))
            y_pred = np.squeeze(self.data["column_scaler"]["close"].inverse_transform(y_hat))
            self.data['test_df']['yhat'] = y_pred
        else:
            pass
        
    def plot_forecasts(self):
        self.data['test_df'].sort_index(inplace=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data['df'].index, y=self.data['df']['close'], name='Actual'))
        fig.add_trace(go.Scatter(x=self.data['test_df'].index, y=self.data['test_df']['yhat'],                                    name='Forecasts'))
        fig.show()

    def get_metrics(self):
        r2 = r2_score(self.data['test_df']['close'], self.data['test_df']['yhat'])
        mae = mean_absolute_error(self.data['test_df']['close'], self.data['test_df']['yhat'])
        mse = mean_squared_error(self.data['test_df']['close'], self.data['test_df']['yhat'], squared=True)
        rmse = mean_squared_error(self.data['test_df']['close'], self.data['test_df']['yhat'], squared=False)
        print("R^2 Score: {}".format(r2))
        print("MAE: {}".format(mae))
        print("MSE: {}".format(mse))
        print("RMSE: {}".format(rmse))
        return [r2, mae, mse, rmse]