#!/usr/bin/env python

import numpy as np
import theano
import theano.tensor as T

# Read the data into a list of words
def read_data(filename):
  # f = zipfile.ZipFile(filename)
  f = open(filename)
  # for name in f.namelist():
  return f.read().split() 
  f.close()

class UnigramLanguageModel(object):
    """
    Unigram language model
    """
    def __init__(self, texts, word2idx):
        """
        Initialize language model with training sentences
        """
        probabilities_values = np.zeros(len(word2idx))
        total = 0

        # text is just a list of words
        for text in texts:
        	
        	for word in text:
        		word_id = word2idx[word]

        	probabilities_values[word_id] += 1
        	total += 1

       	probabilities_values /= total
        self.probabilities = probabilities_values
        self.bins = np.add.accumulate(probabilities_values)
        self.word2idx = word2idx
        self.vocab_size = len(word2idx)

    def likelihood(self, word_ids):
        """
        probability of a tensor of word is under the unigram model.
        """
        return self.probabilities[word_ids]

    def sample(self):
        """
        sample from unigram language model
        """
        # return np.digitize(random_sample(size), self.bins)
        r = np.random.uniform(0, 1)
        s = 0
        for i in xrange(self.vocab_size):
        	s += self.likelihood(i)
        	if s >= r:
        		return i
       	# Just for safe reasons - 3 final values for c_1, c_2, c_3 will be zero anyway
       	return self.vocab_size-3


# A simple skip-gram word2vec model
# It is meant to use with small vocabulary, so the prob is estimated directly using softmax 
class RankingSkipGram(object):

	# Input pos : matrix with size: batch_size * 3
	# Same as Input neg

	def __init__(self, rng, pos_input, neg_input, vocabulary_size, embedding_size=10, hidden_size=0):

		batch_size = pos_input.shape[0]

		# Embedding weight
		E_values = np.asarray(rng.uniform(low=-0.1, high=0.1,
                                                size=(vocabulary_size, embedding_size)),
                                                dtype=theano.config.floatX)
		self.embedding = theano.shared(value=E_values, name='embeddings', borrow=True)

		# emb_input_tmp = self.embedding[batch_input]
		# self.emb_input = emb_input_tmp.reshape((batch_size, -1))

		self.pos_emb = self.embedding[pos_input].reshape((batch_size, -1))
		self.neg_emb = self.embedding[neg_input].reshape((batch_size, -1))

		self.params = [self.embedding]

		last_pos_layer = self.pos_emb
		last_neg_layer = self.neg_emb
		last_layer_size = 3 * embedding_size

		# Hidden Layer

		if hidden_size > 0:

			W_h_values = np.asarray(rng.uniform(low=-0.1, high=0.1, size=(last_layer_size, hidden_size)), dtype=theano.config.floatX)
			self.W_h = theano.shared(value=W_h_values, name='W_h', borrow=True)
			self.b_h = theano.shared(value=np.zeros((hidden_size,), dtype=theano.config.floatX), name='b_h', borrow=True)

			last_pos_layer =  T.tanh(T.dot(last_pos_layer, self.W_h) + self.b_h)
			last_neg_layer =  T.tanh(T.dot(last_neg_layer, self.W_h) + self.b_h)
			last_layer_size = hidden_size

			self.params.append(self.W_h)
			self.params.append(self.b_h)

		# Output Layer

		W_values = np.asarray(rng.uniform(low=-0.1, high=0.1, size=(last_layer_size, 1)), dtype=theano.config.floatX)
		self.W_o = theano.shared(value=W_values, name='W_o', borrow=True)
		self.b_o = theano.shared(value=np.zeros((1,), dtype=theano.config.floatX), name='b_o', borrow=True)
		self.params += [self.W_o, self.b_o]

		self.pos_score = T.dot(last_pos_layer, self.W_o) + self.b_o
		self.neg_score = T.dot(last_neg_layer, self.W_o) + self.b_o

		# self.loss = T.maximum(0, 1 - self.pos_score + self.neg_score)

		# O = np.ones((batch_size, ), dtype=theano.config.floatX)
		# shared_O = theano.shared(value=O, name='ones', borrow=True)

		ones = T.ones_like(self.pos_score)
		difference = ones - self.pos_score + self.neg_score

		self.loss = T.maximum(0, T.mean(difference))

		# # Softmax weight
		# W_values = np.asarray(rng.uniform(low=-0.1, high=0.1, size=(embedding_size, vocabulary_size)), dtype=theano.config.floatX)
		# self.W_sm = theano.shared(value=W_values, name='W', borrow=True)
		# self.b_sm = theano.shared(value=np.zeros((vocabulary_size,), dtype=theano.config.floatX), name='b', borrow=True)
		

		# self.unnorm_prob = T.dot(self.emb_input, self.W_sm) + self.b_sm
		
	
	def normalize_embeddings(self):

		norm = T.sqrt((self.embedding ** 2).sum())

		normalized_embeddings = self.embedding / norm

		return normalized_embeddings


class ProximitySkipGram(object):

	# Input : matrix with size: batch_size * 3 (word, context and collection)
	

	def __init__(self, rng, input, vocabulary_size, embedding_size=10, hidden_size=0):

		batch_size = input.shape[0]

		# Embedding weight
		E_values = np.asarray(rng.uniform(low=-0.1, high=0.1,
                                                size=(vocabulary_size, embedding_size)),
                                                dtype=theano.config.floatX)
		self.embedding = theano.shared(value=E_values, name='embeddings', borrow=True)

		self.emb = self.embedding[input].reshape((batch_size, -1))

		self.params = [self.embedding]

		last_layer = self.emb
		last_layer_size = 3 * embedding_size

		# Hidden Layer

		if hidden_size > 0:

			W_h_values = np.asarray(rng.uniform(low=-0.1, high=0.1, size=(last_layer_size, hidden_size)), dtype=theano.config.floatX)
			self.W_h = theano.shared(value=W_h_values, name='W_h', borrow=True)
			self.b_h = theano.shared(value=np.zeros((hidden_size,), dtype=theano.config.floatX), name='b_h', borrow=True)

			last_layer =  T.tanh(T.dot(last_layer, self.W_h) + self.b_h)
			last_layer_size = hidden_size

			self.params.append(self.W_h)
			self.params.append(self.b_h)

		# Softmax weight: Only yes and no at output ...
		W_values = np.asarray(rng.uniform(low=-0.1, high=0.1, size=(last_layer_size, 2)), dtype=theano.config.floatX)
		self.W_sm = theano.shared(value=W_values, name='W', borrow=True)
		self.b_sm = theano.shared(value=np.zeros((2,), dtype=theano.config.floatX), name='b', borrow=True)
		
		self.unnorm_prob = T.dot(last_layer, self.W_sm) + self.b_sm

		self.params += [self.W_sm, self.b_sm]
		
		self.p_y_given_x = T.nnet.softmax(self.unnorm_prob)
	
	def normalize_embeddings(self):

		norm = T.sqrt((self.embedding ** 2).sum())

		normalized_embeddings = self.embedding / norm

		return normalized_embeddings


	def negative_log_likelihood(self, y):

		loss = -T.mean(T.log2(self.p_y_given_x)[T.arange(y.shape[0]), y])

		return loss



