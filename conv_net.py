import numpy as np
import tensorflow as tf

#this class generates a convolutional neural network with ReLU activations for hidden layers and linear activation for the final layer

#conv_layer array: list of lists containing dimensions of convolutional filters (ex. [3, 3, 1, 4] defines a convolutional filter of width and height 3 that takes an input image with one channel and outputs an image with 4 channels)
#full_layer_array: list of lists containing dimensions of fully connected layers (ex. [500, 300] defines a layer that takes a tensor of size (x, 500) as an input and outputs a tensor of (x, 400))
#input_dims: list defining the dimensions of one example of the convolutional network's input data ([width, height, channels])

class network():
	def __init__(self, conv_layer_array, full_layer_array, input_dims, conv_mode="VALID", pool_func=tf.nn.max_pool):
		self.test_input_val = tf.placeholder(tf.float32, [None, input_dims[0], input_dims[1], input_dims[2]])

		self.conv_weights = []
		self.full_conn_weights = [[None, None]]

		self.conv_mode = conv_mode

		count = 0
		for i in conv_layer_array:
			count += 1
			conv_vars = self.generate_weights_biases(i, name="conv_layer_" + str(count))
			self.conv_weights.append(conv_vars)
		count = 0

        
		sizing_op = self.conv_over(self.test_input_val)
		dim_arr = sizing_op.shape[1:]
		product = 1
		for value in dim_arr:
			product *= int(value)
		self.bridge_dims = [product, full_layer_array[0][0]]
		

		
		self.full_conn_weights[0] = self.generate_weights_biases(self.bridge_dims, name="bridging_layer")

		for j in full_layer_array:
			count += 1
			full_conn_vars = self.generate_weights_biases(j, name="full_conn_layer_" + str(count))
			self.full_conn_weights.append(full_conn_vars)
		

	def generate_weights_biases(self, weights_shape, name):
		weights = tf.Variable(tf.random_normal(weights_shape, stddev=0.1), name=name+"_weights")
		biases = tf.Variable(tf.random_normal([weights_shape[-1]]), name=name+"_biases")
		return [weights, biases]
    
	def conv_over(self, input_val):
		with tf.name_scope("conv_net_forward_pass"):
			last_activ = input_val
			count = 1
			
			for layer in self.conv_weights:
				logits = tf.nn.conv2d(last_activ, layer[0], strides=[1, 1, 1, 1], padding=self.conv_mode)
				logits = tf.add(logits, layer[1])
				logits = tf.nn.relu(logits)
				with tf.name_scope(str(count) + "_max_pooling"):
					pool = tf.nn.max_pool(logits, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")
					last_activ = pool
				count += 1
		return last_activ
	def full_activation(self, input_val):
		conv_output = self.conv_over(input_val)
		conv_output = tf.reshape(conv_output, [-1, self.bridge_dims[0]])
		with tf.name_scope("full_conn_net_forward_pass"):
			last_activ = conv_output
			count = 1

			for i in range(len(self.full_conn_weights) - 1):
				layer = self.full_conn_weights[i]
				mul_value = tf.matmul(last_activ, layer[0])
				add_value = tf.add(mul_value, layer[1])
				activation = tf.nn.relu(add_value)
				last_activ = activation
			
			mul = tf.matmul(last_activ, self.full_conn_weights[-1][0])
			op_add = tf.add(mul, self.full_conn_weights[-1][1])
		return op_add
	
	def network_variables(self):
		op_var = []
		for chip in self.conv_weights:
			op_var.extend(chip)
		for chop in self.full_conn_weights:
			op_var.extend(chop)
		
		return op_var
    

