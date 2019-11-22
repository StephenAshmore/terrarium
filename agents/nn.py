from typing import Any, Mapping, List, Tuple, Optional, Union
import tensorflow as tf
import numpy as np
import random
import sys


# Base class for neural net layers
class Layer(object):
    def __init__(self) -> None:
        self.params: List[tf.Tensor]

    # Repeats a tensor as necessary and crops it to fit the specified output size
    @staticmethod
    def resize(tensor: tf.Tensor, newsize: int) -> tf.Tensor:
        if newsize < tensor.shape[1]:
            return tensor[ : , 0 : newsize]
        elif newsize > tensor.shape[1]:
            multiples = (newsize + int(tensor.shape[1]) - 1) // tensor.shape[1]
            tiled = tf.tile(tensor, [1, multiples])
            if newsize < tiled.shape[1]:
                return tiled[ : , 0 : newsize]
            else:
                return tiled
        else:
            return tensor

    # Gather all the variable values into an object for serialization
    def marshall(self) -> Mapping[str, Any]:
        return { 'params': [ p.numpy().tolist() for p in self.params ] }

    # Load the variables from a deserialized object
    def unmarshall(self, ob: Mapping[str, Any]) -> None:
        params = ob['params']
        if len(params) != len(self.params):
            raise ValueError('Mismatching number of params')
        for i in range(len(params)):
            self.params[i].assign(np.array(params[i]))

    # Returns the number of weights in this layer
    def weightCount(self) -> int:
        wc = 0
        for p in self.params:
            s = 1
            for d in p.shape:
                s *= d
            wc += s
        return wc


# A skip connection for making residual networks
class LayerSkip(Layer):
    def __init__(self) -> None:
        self.params = []

    def act(self, x: tf.Tensor, earlier_source: tf.Tensor) -> tf.Tensor:
        return Layer.resize(earlier_source, x.shape[1]) + x


# A tanh activation function
class LayerTanh(Layer):
    def __init__(self) -> None:
        self.params = []

    def act(self, x: tf.Tensor) -> tf.Tensor:
        return tf.nn.tanh(x)


# A leaky rectifier activation function
class LayerLeakyRectifier(Layer):
	def __init__(self):
		self.params = []

	def act(self, x: tf.Tensor) -> tf.Tensor:
		return tf.nn.leaky_relu(x)



# A tanh activation function
class LayerEl(Layer):
    def __init__(self) -> None:
        self.params = []

    def act(self, x: tf.Tensor) -> tf.Tensor:
        return tf.nn.elu(x)



# A softmax activation function
class LayerSoftmax(Layer):
	def __init__(self):
		self.params = []

	def act(self, x: tf.Tensor) -> tf.Tensor:
		return tf.nn.softmax(x)


# Computes pair-wise products to reduce a vector size by 2
class LayerProductFuser(Layer):
	def __init__(self):
		self.params = []

	def act(self, x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
		return tf.multiply(x1, x2)


# Computes pair-wise products to reduce a vector size by 2
class LayerProductPooling(Layer):
    def __init__(self) -> None:
        self.params = []

    def act(self, x: tf.Tensor) -> tf.Tensor:
        half_size = int(x.shape[1]) // 2
        if int(x.shape[1]) != half_size * 2:
            raise ValueError("Expected an even number of input values")
        two_halves = tf.reshape(x, [-1, 2, half_size])
        return tf.multiply(two_halves[:, 0], two_halves[:, 1])


# Repeats the incoming tensor as many times as necessary, then truncates to the output size
class LayerRepeater(Layer):
	def __init__(self):
		self.params = []

	def act(self, x: tf.Tensor) -> tf.Tensor:
		return Layer.resize(x, outputsize)



# Randomly connects inputs to outputs. (No weights)
class LayerShuffle(Layer):
	def __init__(self, size: int):
		indexes = [i for i in range(incoming.shape[1])]
		random.seed(1234)
		random.shuffle(indexes)
		self.params = []

	def act(self, x: tf.Tensor) -> tf.Tensor:
		return tf.gather(z, indexes, axis = 1)


class LayerMaxPooling2d(Layer):
	def __init__(self):
		self.params = []

	def act(self, x: tf.Tensor) -> tf.Tensor:
		h = x.shape[1]
		w = x.shape[2]
		c = x.shape[3]
		cols = tf.reshape(incoming, (-1, h, int(w) // 2, 2, c))
		halfwidth = tf.math.maximum(cols[:,:,:,0,:], cols[:,:,:,1,:])
		rows = tf.reshape(halfwidth, (-1, int(h) // 2, 2, int(w) // 2, c))
		halfheight = tf.math.maximum(rows[:,:,0,:,:], rows[:,:,1,:,:])
		return tf.reshape(halfheight, (-1, int(h) // 2, int(w) // 2, c))


# A linear (a.k.a. "fully-connected", a.k.a. "dense") layer
class LayerLinear(Layer):
    def __init__(self, inputsize: int, outputsize: int):
        self.weights = tf.Variable(tf.random.normal([inputsize, outputsize], stddev = max(0.03, 1.0 / inputsize), dtype = tf.float32))
        self.bias = tf.Variable(tf.random.normal([outputsize], stddev = max(0.03, 1.0 / inputsize), dtype = tf.float32))
        self.params = [ self.weights, self.bias ]

    def act(self, x: tf.Tensor) -> tf.Tensor:
        return tf.add(tf.matmul(x, self.weights), self.bias)


# The input should be a single integer value
class LayerCatTable(Layer):
    def __init__(self, categories: int, outputsize: int):
        self.outputsize = outputsize
        self.weights = tf.Variable(tf.random.uniform([categories, outputsize], dtype = tf.float32))
        self.params = [ self.weights ]

    def act(self, x: tf.Tensor) -> tf.Tensor:
        return tf.reshape(tf.gather(self.weights, x), (-1, self.outputsize))


# Connects each output unit to a specified number of randomly-selected inputs.
# Also adds the input to make it a residual layer
class LayerRandom(Layer):
	def __init__(self, inputsize: int, outputsize: int, connections: int, reps: int):
		if connections * reps + 1 > inputsize:
			raise ValueError("There are not enough inputs for so many connections")
		self.conns = []
		for i in range(outputsize):
			cands = [x for x in range(inputsize) if x != i]
			random.shuffle(cands)
			self.conns.append(cands[:connections * reps])
		self.weights = tf.Variable(tf.random_normal([outputsize, connections], stddev = 1.0 / (connections * reps), dtype = tf.float64))
		self.bias = tf.Variable(tf.random_normal([outputsize], stddev = 1.0 / (connections * reps), dtype = tf.float64))
        self.params = [ self.weights, self.bias ]

    def act(self, x: tf.Tensor) -> tf.Tensor:
		gathered_inputs = tf.gather(x, self.conns, axis = 1)
		tiled_weights = tf.tile(self.weights, [1, reps])
		prod = tf.multiply(gathered_inputs, tiled_weights)
		unbiased = tf.reduce_sum(prod, 2)
		return unbiased + self.bias + Layer.resize(incoming, outputsize)


# Connects each output unit to inputs with the offsets 1, 2, 4, 8, 16, ...
# Also adds the input to make it a residual layer
class LayerToroid_A(Layer):
	def __init__(self, inputsize: int, outputsize: int):
		if inputsize * 2 < outputsize or outputsize * 2 < inputsize:
			raise ValueError("input and output sizes cannot differ by more than a factor of 2")
		connections_per_output = (inputsize - 1).bit_length()
		self.weights = tf.Variable(tf.random_normal([outputsize, connections_per_output], stddev = 1.0 / connections_per_output, dtype = tf.float64))
		self.bias = tf.Variable(tf.random_normal([outputsize], stddev = 1.0 / connections_per_output, dtype = tf.float64))
        self.params = [ self.weights, self.bias ]

		# Generate nodes for each output
		self.o_indexes = []
		for i in range(outputsize):
			i_indexes = []
			j = 1
			while True:
				if j >= inputsize:
					break
				i_indexes.append((i + j) % inputsize)
				j *= 2
			self.o_indexes.append(i_indexes)

    def act(self, x: tf.Tensor) -> tf.Tensor:
		gathered_inputs = tf.gather(x, self.o_indexes, axis = 1)
		unbiased = tf.math.reduce_sum(gathered_inputs * self.weights, axis = 2)
		without_residual = unbiased + self.bias
		return without_residual + Layer.resize(incoming, outputsize)


# Connects each output unit to inputs with the offsets 1, 2, 4, 8, 16, ...
# Also adds the input to make it a residual layer
class LayerToroid_B(Layer):
	def __init__(self, inputsize: int, outputsize: int):
		if inputsize * 2 < outputsize or outputsize * 2 < inputsize:
			raise ValueError("input and output sizes cannot differ by more than a factor of 2")
		connections_per_output = (inputsize - 1).bit_length()
		self.weights = tf.Variable(tf.random_normal([outputsize, 1], stddev = 1.0 / connections_per_output, dtype = tf.float64))
		self.bias = tf.Variable(tf.random_normal([outputsize], stddev = 1.0 / connections_per_output, dtype = tf.float64))
        self.params = [ self.weights, self.bias ]

		# Generate nodes for each output
		self.o_indexes = []
		for i in range(outputsize):
			i_indexes = []
			j = 1
			while True:
				if j >= inputsize:
					break
				i_indexes.append((i + j) % inputsize)
				j *= 2
			self.o_indexes.append(i_indexes)

    def act(self, x: tf.Tensor) -> tf.Tensor:
		gathered_inputs = tf.gather(x, self.o_indexes, axis = 1)
		tiled_weights = tf.tile(self.weights, [1, connections_per_output])
		unbiased = tf.math.reduce_sum(gathered_inputs * tiled_weights, axis = 2)
		without_residual = unbiased + self.bias
		return without_residual + Layer.resize(incoming, outputsize)


# Connects each output unit to inputs with the offsets 1, 2, 4, 8, 16, ...
# Also adds the input to make it a residual layer
class LayerToroid_C(Layer):
	def __init__(self, inputsize: int, outputsize: int):
		if inputsize * 2 < outputsize or outputsize * 2 < inputsize:
			raise ValueError("input and output sizes cannot differ by more than a factor of 2")
		connections_per_output = (inputsize - 1).bit_length()
		self.weights = tf.Variable(tf.random_normal([1, connections_per_output], stddev = 1.0 / connections_per_output, dtype = tf.float64))
		self.bias = tf.Variable(tf.random_normal([outputsize], stddev = 1.0 / connections_per_output, dtype = tf.float64))
        self.params = [ self.weights, self.bias ]

		# Generate nodes for each output
		self.o_indexes = []
		for i in range(outputsize):
			i_indexes = []
			j = 1
			while True:
				if j >= inputsize:
					break
				i_indexes.append((i + j) % inputsize)
				j *= 2
			self.o_indexes.append(i_indexes)

    def act(self, x: tf.Tensor) -> tf.Tensor:
		gathered_inputs = tf.gather(x, self.o_indexes, axis = 1)
		tiled_weights = tf.tile(self.weights, [outputsize, 1])
		unbiased = tf.math.reduce_sum(gathered_inputs * tiled_weights, axis = 2)
		without_residual = unbiased + self.bias
		return without_residual + Layer.resize(incoming, outputsize)


# A layer that connects each unit to log_2(n) other units, according to the edges in a hypercube
class LayerHypercube_A(Layer):
	# Let N = batch samples, F = feature maps, B = bits, I = inputs, O = outputs
	# incoming has shape=(N,I)
	def __init__(self, inputsize: int, outputsize: int, featuremaps: int):
		if inputsize > 2 * outputsize or outputsize > 2 * inputsize:
			raise ValueError("The inputs and outputs may not differ by more than a factor of 2")
		bits = (int(incoming.shape[1]) - 1).bit_length()
		self.weights = tf.Variable(tf.random_normal([featuremaps, bits, outputsize], stddev = 0.5 / bits, dtype = tf.float64)) # shape=(F,B,O)
		self.bias = tf.Variable(tf.random_normal([featuremaps, outputsize], stddev = 0.5 / bits, dtype = tf.float64)) # shape=(F,O)
        self.params = [ self.weights, self.bias ]

		# Regroup all the inputs to align with the edges in the hypercube
		self.fm = [] # Construct shape=(F,B,O)
		for k in range(featuremaps):
			outer = []
			for j in range(bits):
				inner = []
				for i in range(outputsize):
					inner.append((i ^ (1 << j)) % inputsize)
				outer.append(inner)
			self.fm.append(outer)

    def act(self, x: tf.Tensor) -> tf.Tensor:
		inputs_aligned_with_weights = tf.gather(x, self.fm, axis = 1) # shape=(N,F,B,O)

		# Multiply by the weights and add the bias
		prod = tf.multiply(inputs_aligned_with_weights, self.weights) # shape=(N,F,B,O)
		unbiased_net = tf.reduce_sum(prod, axis = 2) # shape=(N,F,O)
		biased_net = tf.add(unbiased_net, self.bias) # shape=(N,F,O)
		reshaped_biased_net = tf.reshape(biased_net, [-1, featuremaps * outputsize]) # shape=(N,F*O)

		# Make it residual by adding the input to the output
		tiled_incoming = tf.tile(Layer.resize(incoming, outputsize), [1, featuremaps]) # shape=(N,F*O)
		return tf.add(reshaped_biased_net, tiled_incoming) # shape=(N,F*O)


# A layer that connects each unit to log_2(n) other units, according to the edges in a hypercube
class LayerHypercube_B(Layer): # Shares weights over bits
	# Let N = batch samples, F = feature maps, B = bits, V = vertices
	# incoming has shape=(N,I)
	def __init__(self, inputsize: int, outputsize: int, featuremaps: int):
		if inputsize > 2 * outputsize or outputsize > 2 * inputsize:
			raise ValueError("The inputs and outputs may not differ by more than a factor of 2")
		bits = (int(incoming.shape[1]) - 1).bit_length()
		self.weights = tf.Variable(tf.random_normal([featuremaps, 1, outputsize], stddev = 0.5 / bits, dtype = tf.float64)) # shape=(F,1,O)
		self.bias = tf.Variable(tf.random_normal([featuremaps, outputsize], stddev = 0.5 / bits, dtype = tf.float64)) # shape=(F,O)
        self.params = [ self.weights, self.bias ]

		# Regroup all the inputs to align with the edges in the hypercube
		self.fm = [] # Construct shape=(F,B,O)
		for k in range(featuremaps):
			outer = []
			for j in range(bits):
				inner = []
				for i in range(outputsize):
					inner.append((i ^ (1 << j)) % inputsize)
				outer.append(inner)
			self.fm.append(outer)

    def act(self, x: tf.Tensor) -> tf.Tensor:
		inputs_aligned_with_weights = tf.gather(x, self.fm, axis = 1) # shape=(N,F,B,O)

		# Multiply by the weights and add the bias
		tiled_weights = tf.tile(self.weights, [1, bits, 1]) # shape=(F,B,O)
		prod = tf.multiply(inputs_aligned_with_weights, tiled_weights) # shape=(N,F,B,O)
		unbiased_net = tf.reduce_sum(prod, axis = 2) # shape=(N,F,O)
		biased_net = tf.add(unbiased_net, self.bias) # shape=(N,F,O)
		reshaped_biased_net = tf.reshape(biased_net, [-1, featuremaps * outputsize]) # shape=(N,F*O)

		# Make it residual by adding the input to the output
		tiled_incoming = tf.tile(Layer.resize(incoming, outputsize), [1, featuremaps]) # shape=(N,F*O)
		return tf.add(reshaped_biased_net, tiled_incoming) # shape=(N,F*O)


# A layer that connects each unit to log_2(n) other units, according to the edges in a hypercube
class LayerHypercube_C(Layer): # Shares weights over vertices
	# Let N = batch samples, F = feature maps, B = bits, V = vertices
	# incoming has shape=(N,I)
	def __init__(self, inputsize: int, outputsize: int, featuremaps: int):
		if inputsize > 2 * outputsize or outputsize > 2 * inputsize:
			raise ValueError("The inputs and outputs may not differ by more than a factor of 2")
		bits = (int(incoming.shape[1]) - 1).bit_length()
		self.weights = tf.Variable(tf.random_normal([featuremaps, bits, 1], stddev = 0.5 / bits, dtype = tf.float64)) # shape=(F,B,1)
		self.bias = tf.Variable(tf.random_normal([featuremaps, 1], stddev = 0.5 / bits, dtype = tf.float64)) # shape=(F,1)
        self.params = [ self.weights, self.bias ]

		# Regroup all the inputs to align with the edges in the hypercube
		self.fm = [] # Construct shape=(F,B,O)
		for k in range(featuremaps):
			outer = []
			for j in range(bits):
				inner = []
				for i in range(outputsize):
					inner.append((i ^ (1 << j)) % inputsize)
				outer.append(inner)
			self.fm.append(outer)

    def act(self, x: tf.Tensor) -> tf.Tensor:
		inputs_aligned_with_weights = tf.gather(x, self.fm, axis = 1) # shape=(N,F,B,O)

		# Multiply by the weights and add the bias
		tiled_weights = tf.tile(self.weights, [1, 1, outputsize]) # shape=(F,B,O)
		prod = tf.multiply(inputs_aligned_with_weights, tiled_weights) # shape=(N,F,B,O)
		unbiased_net = tf.reduce_sum(prod, axis = 2) # shape=(N,F,O)
		tiled_bias = tf.tile(self.bias, [1, outputsize]) # shape=(F,O)
		biased_net = tf.add(unbiased_net, tiled_bias) # shape=(N,F,O)
		reshaped_biased_net = tf.reshape(biased_net, [-1, featuremaps * outputsize]) # shape=(N,F*O)

		# Make it residual by adding the input to the output
		tiled_incoming = tf.tile(Layer.resize(incoming, outputsize), [1, featuremaps]) # shape=(N,F*O)
		return tf.add(reshaped_biased_net, tiled_incoming) # shape=(N,F*O)


class LayerConv(Layer):
    # filter_shape should take the form: (height, width, channels_incoming, channels_outgoing)
	def __init__(self, filter_shape: Tuple[int, ...]):
		spatial_size = 1
		for i in range(0, len(filter_shape) - 2):
			spatial_size *= filter_shape[i]
		self.weights = tf.Variable(tf.random_normal(filt_shape, stddev = 1.0 / spatial_size, dtype = tf.float64))
        self.params = [ self.weights ]

    def act(self, x: tf.Tensor) -> tf.Tensor:
		self.activation = tf.nn.convolution(x, self.weights, "SAME")
