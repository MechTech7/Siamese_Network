import conv_net as cnn
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)


input_1 = tf.placeholder(tf.float32, [None, 28, 28, 1], name="input_example_2")
input_2 = tf.placeholder(tf.float32, [None, 28, 28, 1], name="input_example_2")
equals_labels = tf.placeholder(tf.int32, [None], name="equals_labels")

nt = cnn.network(conv_layer_array=[[3, 3, 1, 4], [3, 3, 4, 8], [3, 3, 8, 8]], full_layer_array=[[500, 500], [500, 5]], input_dims=[28, 28, 1])

logits_1 = nt.full_activation(input_1)
logits_2 = nt.full_activation(input_2)

euc_dist = tf.norm(logits_1-logits_2, ord='euclidean', axis=-1)


with tf.name_scope("contrastive_loss"):
    cont_loss = tf.contrib.losses.metric_learning.contrastive_loss(equals_labels, logits_1, logits_2, margin=1.3)

optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cont_loss, var_list=nt.network_variables())
init_op = tf.global_variables_initializer()

steps = 4001
batch_size = 50

with tf.Session() as sesh:
    sesh.run(init_op)

    for i in range(steps):
        batch_1, batch_1_label = mnist.train.next_batch(batch_size)
        batch_1 = np.reshape(batch_1, [-1, 28, 28, 1])

        batch_2, batch_2_label = mnist.train.next_batch(batch_size)
        batch_2 = np.reshape(batch_2, [-1, 28, 28, 1])

        eq_label = np.equal(batch_1_label, batch_2_label)
        _, ls = sesh.run([optimizer, cont_loss], feed_dict={input_1: batch_1, input_2: batch_2, equals_labels: eq_label})

        if i % 50 == 0:
            print(ls)

    test_1, test_1_l = mnist.test.next_batch(10)
    test_1 = np.reshape(test_1, [-1, 28, 28, 1])

    test_2, test_2_l = mnist.test.next_batch(10)
    test_2 = np.reshape(test_2, [-1, 28, 28, 1])

    test_eq = np.equal(test_1_l, test_2_l)

    print("first_batch labels: " + str(test_1_l))
    print("second_batch labels: " + str(test_2_l))
    print("equals_label: " + str(test_eq))

    dist = sesh.run(euc_dist, feed_dict={input_1: test_1, input_2: test_2, equals_labels: test_eq})
    print(dist)

#So, the siamese netowrk actually works to some extent, it 