import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class BasicDeepNeuralNetwork:
    def __init__(self):
        # This part will import data into the network.

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

        x_scaled_training = x_scaler.fit_transform(x_training)
        y_scaled_training = y_scaler.fit_transform(y_training)

        x_scaled_testing = x_scaler.transform(x_testing)
        y_scaled_testing = y_scaler.transform(y_testing)

        print(x_scaled_testing.shape)
        print(y_scaled_testing.shape)

        print('Y values were scaled by multiplying by {:.10f} and adding {:.4f}'.format(y_scaler.scale_[0],
                                                                                        y_scaler.min_[0]))

        # Define model parameters
        learning_rate = 0.001
        training_epochs = 100
        display_step = 5

        number_of_inputs = x_scaled_testing.shape[1]
        number_of_outputs = 1

        layer_1_nodes = 50
        layer_2_nodes = 100
        layer_3_nodes = 50

        # Input layer
        with tf.variable_scope('input'):
            x = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

        # Layer 1
        with tf.variable_scope('layer_1'):
            weights = tf.get_variable(name='weights1', shape=[number_of_inputs, layer_1_nodes],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable(name='biases1', shape=[layer_1_nodes], initializer=tf.zeros_initializer())
            layer_1_output = tf.nn.relu(tf.matmul(x, weights) + biases)

        # Layer 2
        with tf.variable_scope('layer_2'):
            weights = tf.get_variable(name='weights2', shape=[layer_1_nodes, layer_2_nodes],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable(name='biases2', shape=[layer_2_nodes], initializer=tf.zeros_initializer())
            layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

        # Layer 3
        with tf.variable_scope('layer_3'):
            weights = tf.get_variable(name='weights3', shape=[layer_2_nodes, layer_3_nodes],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable(name='biases3', shape=[layer_3_nodes], initializer=tf.zeros_initializer())
            layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

        # Output Layer
        with tf.variable_scope('output'):
            weights = tf.get_variable(name='weights4', shape=[layer_3_nodes, number_of_outputs],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable(name='biases4', shape=[number_of_outputs], initializer=tf.zeros_initializer())
            prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)

        # Training. A cost function is need to train a network
        with tf.variable_scope('cost'):
            y = tf.placeholder(tf.float32, shape=(None, 1))
            cost = tf.reduce_mean(tf.squared_difference(prediction, y))

        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        with tf.variable_scope('logging'):
            tf.summary.scalar('current_cost', cost)
            summary = tf.summary.merge_all()

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            training_writer = tf.summary.FileWriter('../logs/training', session.graph)
            testing_writer = tf.summary.FileWriter('../logs/testing', session.graph)

            # Training loop
            for epoch in range(training_epochs):
                session.run(optimizer, feed_dict={x: x_scaled_training, y: y_scaled_training})

                if epoch % display_step == 0:
                    training_cost, training_summary = session.run([cost, summary], feed_dict={x: x_scaled_training,
                                                                                              y: y_scaled_training})
                    testing_cost, testing_summary = session.run([cost, summary], feed_dict={x: x_scaled_testing,
                                                                                            y: y_scaled_testing})

                    training_writer.add_summary(training_summary, epoch)
                    testing_writer.add_summary(testing_summary, epoch)

                    print(epoch, training_cost, testing_cost)

            print('Training complete')

            final_training_cost = session.run(cost, feed_dict={x: x_scaled_training, y: y_scaled_training})
            final_testing_cost = session.run(cost, feed_dict={x: x_scaled_testing, y: y_scaled_testing})

            print('Final Training Cost: {}'.format(final_training_cost))
            print('Final Testing Cost: {}'.format(final_testing_cost))


basicNeuralNetwork = BasicDeepNeuralNetwork()
