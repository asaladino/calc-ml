import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class BasicDeepNeuralNetwork:
    def __init__(self):
        # This part will import data into the network.
        pass

    def load_and_scale_data(self):
        # Load training data set from csv file
        training_data_df = pd.read_csv('../data/training_data.csv', dtype=float, index_col='id')

        self.x_training = training_data_df.drop(['solution'], axis=1).values
        self.y_training = training_data_df[['solution']].values

        # Load testing data set from csv file
        self.testing_data_df = pd.read_csv('../data/testing_data.csv', dtype=float, index_col='id')

        self.x_testing = self.testing_data_df.drop(['solution'], axis=1).values
        self.y_testing = self.testing_data_df[['solution']].values

        self.x_scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))

        self.x_scaled_training = self.x_scaler.fit_transform(self.x_training)
        self.y_scaled_training = self.y_scaler.fit_transform(self.y_training)

        self.x_scaled_testing = self.x_scaler.transform(self.x_testing)
        self.y_scaled_testing = self.y_scaler.transform(self.y_testing)

        print(self.x_scaled_testing.shape)
        print(self.y_scaled_testing.shape)

        print('Y values were scaled by multiplying by {:.10f} and adding {:.4f}'.format(self.y_scaler.scale_[0],
                                                                                        self.y_scaler.min_[0]))

    def build_layers_keras(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(10000, 3)),
            tf.keras.layers.Dense(50, activation=tf.nn.relu),
            tf.keras.layers.Dense(100, activation=tf.nn.relu),
            tf.keras.layers.Dense(50, activation=tf.nn.relu),
            tf.keras.layers.Dense(1, activation=tf.nn.relu)
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(self.x_scaled_training, self.y_scaled_training, epochs=100)
        model.evaluate(self.x_scaled_testing, self.y_scaled_testing)


basicNeuralNetwork = BasicDeepNeuralNetwork()
basicNeuralNetwork.load_and_scale_data()
basicNeuralNetwork.build_layers_keras()
