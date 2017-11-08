import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load training data set from csv file
training_data_df = pd.read_csv('../data/training_data.csv', dtype=float, index_col='id')

X_training = training_data_df.drop(['solution', 'operator'], axis=1).values
Y_training = training_data_df[['solution']].values

# Load testing data set from csv file
testing_data_df = pd.read_csv('../data/testing_data.csv', dtype=float, index_col='id')

X_testing = testing_data_df.drop(['solution', 'operator'], axis=1).values
Y_testing = testing_data_df[['solution']].values

X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)

print(X_scaled_testing.shape)
print(Y_scaled_testing.shape)

print('Y values were scaled by multiplying by {:.10f} and adding {:.4f}'.format(Y_scaler.scale_[0], Y_scaler.min_[0]))

# Define model parameters
learning_rate = 0.001
training_epochs = 100
display_step = 5

number_of_inputs = X_scaled_testing.shape[1]
number_of_outputs = 1

layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

# Input layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

# Layer 1
with tf.variable_scope('layer_1'):
    weights = tf.get_variable(name='weights1', shape=[number_of_inputs, layer_1_nodes],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases1', shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

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
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    # Training loop
    for epoch in range(training_epochs):
        session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
        
        if epoch % display_step == 0:
            training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
            testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})
            print(epoch, training_cost, testing_cost)

    print('Training complete')

    final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
    final_testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})

    print('Final Training Cost: {}'.format(final_training_cost))
    print('Final Testing Cost: {}'.format(final_testing_cost))
