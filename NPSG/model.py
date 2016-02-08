#!/usr/bin/env python

import numpy as np
import theano
import theano.tensor as T

# A simple skip-gram word2vec model
# It is meant to use with small vocabulary, so the prob is estimated directly using softmax 
class SkipGramW2V(object):

	def __init__(self, rng, batch_input, single_input, vocabulary_size, embedding_size=10):

		batch_size = batch_input.shape[0]


		# Embedding weight
		E_values = np.asarray(rng.uniform(low=-0.1, high=0.1,
                                                size=(vocabulary_size, embedding_size)),
                                                dtype=theano.config.floatX)
		self.embedding = theano.shared(value=E_values, name='embeddings', borrow=True)

		emb_input_tmp = self.embedding[batch_input]
		self.emb_input = emb_input_tmp.reshape((batch_size, -1))

		# Softmax weight
		W_values = np.asarray(rng.uniform(low=-0.1, high=0.1, size=(embedding_size, vocabulary_size)), dtype=theano.config.floatX)
		self.W_sm = theano.shared(value=W_values, name='W', borrow=True)
		self.b_sm = theano.shared(value=np.zeros((vocabulary_size,), dtype=theano.config.floatX), name='b', borrow=True)
		self.unnorm_prob = T.dot(self.emb_input, self.W_sm) + self.b_sm
		
		# prob of context given word
		self.p_c_given_w = T.nnet.softmax(self.unnorm_prob)

		self.params = [self.embedding, self.W_sm, self.b_sm]


		single_emb = self.embedding[single_input]

		# For single input
		self.p_c_given_sw = T.nnet.softmax(T.dot(single_emb, self.W_sm) + self.b_sm)

	# Loss function
	def negative_log_likelihood(self, y):

		loss = -T.mean(T.log2(self.p_c_given_w)[T.arange(y.shape[0]), y])

		return loss

	def predict(self, y):

		prob = self.p_c_given_w[T.arange(y.shape[0]), y]
		return prob

	def normalize_embeddings(self):

		norm = T.sqrt((self.embedding ** 2).sum())

		normalized_embeddings = self.embedding / norm

		return normalized_embeddings



class CollectionSkipGramW2V(object):

	def __init__(self, rng, batch_input, single_input, vocabulary_size, embedding_size=10, hidden_size=[]):

		batch_size = batch_input.shape[0]


		# Embedding weight
		E_values = np.asarray(rng.uniform(low=-0.1, high=0.1,
                                                size=(vocabulary_size, embedding_size)),
                                                dtype=theano.config.floatX)
		self.embedding = theano.shared(value=E_values, name='embeddings', borrow=True)

		self.params = [self.embedding]

		emb_input_tmp = self.embedding[batch_input]
		self.emb_input = emb_input_tmp.reshape((batch_size, -1))

		last_layer = self.emb_input
		last_layer_size = 2 * embedding_size

		self.W_h = dict()
		self.b_h = dict()
		self.L2_norm = 0

		if len(hidden_size) > 0:

			for h in xrange(len(hidden_size)):
				W_h_values = np.asarray(rng.uniform(low=-0.1, high=0.1, size=(last_layer_size, hidden_size[h])), dtype=theano.config.floatX)
				self.W_h[h] = theano.shared(value=W_h_values, name='W_h', borrow=True)
				self.b_h[h] = theano.shared(value=np.zeros((hidden_size[h],), dtype=theano.config.floatX), name='b_h', borrow=True)
				last_layer =   T.tanh(T.dot(last_layer, self.W_h[h]) + self.b_h[h])
				last_layer_size = hidden_size[h]

				self.params.append(self.W_h[h])
				self.params.append(self.b_h[h])
				
				self.L2_norm += (self.W_h[h] ** 2).sum()

		# Softmax weight
		W_values = np.asarray(rng.uniform(low=-0.1, high=0.1, size=(last_layer_size, vocabulary_size-3)), dtype=theano.config.floatX)
		self.W_sm = theano.shared(value=W_values, name='W', borrow=True)
		self.b_sm = theano.shared(value=np.zeros((vocabulary_size-3,), dtype=theano.config.floatX), name='b', borrow=True)
		self.unnorm_prob = T.dot(last_layer, self.W_sm) + self.b_sm

		self.L2_norm += (self.W_sm ** 2).sum()
		
		# prob of context given word
		self.p_c_given_w = T.nnet.softmax(self.unnorm_prob)

		self.params += [self.W_sm, self.b_sm]


		# For single input
		single_emb = self.embedding[single_input].reshape((1, -1))

		last_layer = single_emb
		last_layer_size = 2 * embedding_size



		if len(hidden_size) > 0:

			for h in xrange(len(hidden_size)):
				last_layer =   T.tanh(T.dot(last_layer, self.W_h[h]) + self.b_h[h])
				last_layer_size = hidden_size[h]

		self.single_coded_vector = last_layer

		self.p_c_given_sw = T.nnet.softmax(T.dot(self.single_coded_vector, self.W_sm) + self.b_sm)
				
		# if hidden_size == 0:
		# 	self.p_c_given_sw = T.nnet.softmax(T.dot(single_emb, self.W_sm) + self.b_sm)
		# 	self.single_coded_vector = single_emb
		# else:
		# 	single_h = T.tanh(T.dot(single_emb, self.W_h) + self.b_h)
		# 	self.single_coded_vector = single_h
		# 	self.p_c_given_sw = T.nnet.softmax(T.dot(single_h, self.W_sm) + self.b_sm)

	# Loss function
	def negative_log_likelihood(self, y):

		loss = -T.mean(T.log2(self.p_c_given_w)[T.arange(y.shape[0]), y])


		return loss

	def predict(self, y):

		prob = self.p_c_given_w[T.arange(y.shape[0]), y]
		return prob

	def normalize_embeddings(self):

		norm = T.sqrt((self.embedding ** 2).sum())

		normalized_embeddings = self.embedding / norm

		return normalized_embeddings