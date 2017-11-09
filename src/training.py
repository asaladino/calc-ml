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

        x_training = training_data_df.drop(['solution', 'operator'], axis=1).values
        y_training = training_data_df[['solution']].values

        # Load testing data set from csv file
        testing_data_df = pd.read_csv('../data/testing_data.csv', dtype=float, index_col='id')

        x_testing = testing_data_df.drop(['solution', 'operator'], axis=1).values
        y_testing = testing_data_df[['solution']].values

        x_scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaler = MinMaxScaler(feature_range=(0, 1))

        self.x_scaled_training = x_scaler.fit_transform(x_training)
        self.y_scaled_training = y_scaler.fit_transform(y_training)

        self.x_scaled_testing = x_scaler.transform(x_testing)
        self.y_scaled_testing = y_scaler.transform(y_testing)

        print(self.x_scaled_testing.shape)
        print(self.y_scaled_testing.shape)

        print('Y values were scaled by multiplying by {:.10f} and adding {:.4f}'.format(y_scaler.scale_[0],
                                                                                        y_scaler.min_[0]))

    def load_params(self):
        # Define model parameters
        self.learning_rate = 0.001
        self.training_epochs = 100
        self.display_step = 5

        self.number_of_inputs = self.x_scaled_testing.shape[1]
        self.number_of_outputs = 1

        self.layer_1_nodes = 50
        self.layer_2_nodes = 100
        self.layer_3_nodes = 50

    def build_layers(self):
        # Input layer
        with tf.variable_scope('input'):
            self.x = tf.placeholder(tf.float32, shape=(None, self.number_of_inputs))

        # Layer 1
        with tf.variable_scope('layer_1'):
            weights = tf.get_variable(name='weights1', shape=[self.number_of_inputs, self.layer_1_nodes],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable(name='biases1', shape=[self.layer_1_nodes], initializer=tf.zeros_initializer())
            layer_1_output = tf.nn.relu(tf.matmul(self.x, weights) + biases)

        # Layer 2
        with tf.variable_scope('layer_2'):
            weights = tf.get_variable(name='weights2', shape=[self.layer_1_nodes, self.layer_2_nodes],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable(name='biases2', shape=[self.layer_2_nodes], initializer=tf.zeros_initializer())
            layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

        # Layer 3
        with tf.variable_scope('layer_3'):
            weights = tf.get_variable(name='weights3', shape=[self.layer_2_nodes, self.layer_3_nodes],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable(name='biases3', shape=[self.layer_3_nodes], initializer=tf.zeros_initializer())
            layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

        # Output Layer
        with tf.variable_scope('output'):
            weights = tf.get_variable(name='weights4', shape=[self.layer_3_nodes, self.number_of_outputs],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable(name='biases4', shape=[self.number_of_outputs], initializer=tf.zeros_initializer())
            prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)

        # Training. A cost function is need to train a network
        with tf.variable_scope('cost'):
            self.y = tf.placeholder(tf.float32, shape=(None, 1))
            self.cost = tf.reduce_mean(tf.squared_difference(prediction, self.y))

        with tf.variable_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        with tf.variable_scope('logging'):
            tf.summary.scalar('current_cost', self.cost)
            self.summary = tf.summary.merge_all()

    def train(self):
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            training_writer = tf.summary.FileWriter('../logs/training', session.graph)
            testing_writer = tf.summary.FileWriter('../logs/testing', session.graph)

            # Training loop
            for epoch in range(self.training_epochs):
                session.run(self.optimizer, feed_dict={self.x: self.x_scaled_training, self.y: self.y_scaled_training})

                if epoch % self.display_step == 0:
                    training_cost, training_summary = session.run([self.cost, self.summary], feed_dict={
                        self.x: self.x_scaled_training, self.y: self.y_scaled_training})
                    testing_cost, testing_summary = session.run([self.cost, self.summary], feed_dict={
                        self.x: self.x_scaled_testing, self.y: self.y_scaled_testing})

                    training_writer.add_summary(training_summary, epoch)
                    testing_writer.add_summary(testing_summary, epoch)

                    print(epoch, training_cost, testing_cost)

            print('Training complete')

            final_training_cost = session.run(self.cost, feed_dict={
                self.x: self.x_scaled_training, self.y: self.y_scaled_training})
            final_testing_cost = session.run(self.cost, feed_dict={
                self.x: self.x_scaled_testing, self.y: self.y_scaled_testing})

            print('Final Training Cost: {}'.format(final_training_cost))
            print('Final Testing Cost: {}'.format(final_testing_cost))


basicNeuralNetwork = BasicDeepNeuralNetwork()
basicNeuralNetwork.load_and_scale_data()
basicNeuralNetwork.load_params()
basicNeuralNetwork.build_layers()
basicNeuralNetwork.train()
