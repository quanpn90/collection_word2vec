import collections
import math
from time import time
import numpy as np
import os
import random
from random import shuffle
import theano
import theano.tensor as T
import zipfile
from model import SkipGramW2V, CollectionSkipGramW2V
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Read the data into a list of words
def read_data(filename):
  # f = zipfile.ZipFile(filename)
  f = open(filename)
  # for name in f.namelist():
  return f.read().split() 
  f.close()

def build_dataset(root, text_sources):

	word2idx = dict()
	# In this model, we add 3 "collection" vectors to learn with the word embeddings in the same time
	

	for i, text_source in enumerate(text_sources):
		text_file = os.path.join(root, text_source + ".txt")
		words = read_data(text_file)
		
		for word in words:
			if word not in word2idx:
				word2idx[word] = len(word2idx)

	word2idx["c_1"] = len(word2idx)
	word2idx["c_2"] = len(word2idx)
	word2idx["c_3"] = len(word2idx)

	idx2word = dict(zip(word2idx.values(), word2idx.keys()))	
	# print idx2word[36]
	print "Vocabulary size:", len(word2idx), len(idx2word)
	return word2idx, idx2word

# Extract (word, context) pair list from a set of text sources
# Convert the word ids into tensors to feed into Theano
def extract_skip_grams(root, text_sources, word2idx, skip_window=3, shuffling=True):

	data = list()
	for c, text_source in enumerate(text_sources):

		text_file = os.path.join(root, text_source + ".txt")
		words = read_data(text_file)

		# Scan the word stream to extract (word, context) pairs
		for i, word in enumerate(words):
			inputs = list()
			
			inputs.append(word2idx[word])
			context_word = "c_" + str(c+1)
			inputs.append(word2idx[context_word])


			
			left_context = list()
			right_context = list()

			for j in xrange(skip_window):
				if i - j -1 > 0 :
				  left_context.append(word2idx[words[i - j - 1]])

			for j in xrange(skip_window):
				if i + j + 1 < len(words):
				  right_context.append(word2idx[words[i + j + 1]])


			contexts = left_context + right_context

			for context in contexts:
				data.append([inputs, context])
	
	if shuffling == True:
		np.random.seed(2016)
		print("Shuffling data")
		shuffle(data)

	# Extract the train and label tensors
	train_input = list()
	train_label = list()

	print("Converting list to tensors")
	for w_c in data:
		train_input.append(w_c[0])
		train_label.append(w_c[1])

	input_tensors = np.asarray(train_input, dtype=np.int32)
	label_tensors = np.asarray(train_label, dtype=np.int32)

	return input_tensors, label_tensors




def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
  plt.savefig(filename)


def main():

	# Dataset path
	root = "../art-data"
	text_sources = ["mixture1", "mixture2", "mixture3"]
	np.random.seed(2056)

	word2idx, idx2word = build_dataset(root, text_sources)

	# shuffling=False
	input_tensors, label_tensors = extract_skip_grams(root, text_sources, word2idx) 

	# Model parameters
	vocab_size = len(word2idx)
	embedding_size = 100
	rng = np.random.RandomState(2016)
	learning_rate = 0.1
	batch_size = 4
	n_iter = 2
	hidden_size = [150, 100]
	weight_decay = 3e-5

	# Build model
	print "Compiling Theano Graph ... "
	x_batch = T.imatrix() # Change to imatrix if input has more than 1 word 
	y_batch = T.ivector()
	valid = T.ivector()
	x = T.ivector() # non-batch tensor input
	index = T.lscalar()
	model = CollectionSkipGramW2V(rng, x_batch, x, vocab_size, embedding_size, hidden_size=hidden_size) 

	loss = model.negative_log_likelihood(y_batch) + weight_decay * model.L2_norm
	
	gparams = [T.grad(loss, param) for param in model.params] 

	updates = [(param, param - learning_rate * gparam)
                        for param, gparam in zip(model.params, gparams)] 
	
	shared_x = theano.shared(np.asarray(input_tensors, dtype=theano.config.floatX),
                                 borrow=True)
	shared_y = theano.shared(np.asarray(label_tensors, dtype=theano.config.floatX),
	                                     borrow=True)

	shared_y = T.cast(shared_y, 'int32')
	shared_x = T.cast(shared_x, 'int32')

	data_length = input_tensors.shape[0]
	batch_interval = slice(index * batch_size, (index + 1) * batch_size)

	normalized_embeddings = model.normalize_embeddings()

	# For nearest vectors
	valid_vectors = normalized_embeddings[valid]
	similarity = T.dot(valid_vectors, normalized_embeddings[:-3,].T) 

	

	# Start training now
	valid_examples = np.array(random.sample(np.arange(vocab_size-3), vocab_size-3), dtype=np.int32)
	test_examples = np.array(random.sample(np.arange(vocab_size-3), vocab_size-3), dtype=np.int32)

	# Most corresponding vectors: we need to get the softmax_ed distribution
	p_test = model.p_c_given_sw

	
	# Compiling the train function here
	train_model = theano.function(inputs=[index], # batch index
	                              outputs=loss,
	                              updates=updates,
	                              givens={x_batch: shared_x[batch_interval],
	                                      y_batch: shared_y[batch_interval]}
	                              )



	# A function to compute P(context | word, collection)
	test_model = theano.function(inputs = [x], outputs=p_test)


	# A function to computer sim(word1, word2)
	valid_model = theano.function(inputs=[valid], outputs=similarity)

	# A function to get the coded vector of x = (word, collection). 
	get_embedding = theano.function(inputs=[x], outputs = model.single_coded_vector)

	num_steps = data_length // batch_size
	print "Total training step:", num_steps * n_iter

	for iter in xrange(n_iter):

		print "Start training iter", iter
		total_loss = 0 
		for index in xrange(num_steps):
			train_loss = train_model(index)
			total_loss += train_loss

			# Print loss during training. At least we want to see loss decrease
			if index % 50000 == 0:
			  if index > 0: 
			    average_loss = total_loss / (index + 1)
			    print "Average loss at step ", index, "/", num_steps, ": ", average_loss

			# Print k-nearest words for some valid examples
			if index % 500000  == 0 or index == num_steps - 1:
			  sim = valid_model(valid_examples)
			  for i in xrange(vocab_size-3):
			    valid_word = idx2word[valid_examples[i]]
			    top_k = 8 # number of nearest neighbors
			    nearest = (-sim[i, :]).argsort()[1:top_k+1]
			    log_str = "Nearest to %s:" % valid_word
			    for k in xrange(top_k):
			      close_word = idx2word[nearest[k]]
			      if "_" not in close_word: 
			      	log_str = "%s %s," % (log_str, close_word)
			    print(log_str)

			# Check co-appearance in collections of several test words (currently only Iran...)
			if ( index % 1000000 == 0 and index > 0 ) or index == num_steps - 1:
				
				for i in xrange(3):

					context_word = "c_" + str(i+1)
					test_word = 'Iran'
					test = [word2idx['Iran'], word2idx[context_word]]
					tensor = np.asarray(test, dtype=np.int32)

					p_y_given_w_test = test_model(tensor)[0]
					
					top_k = 8
					closest = np.argsort(p_y_given_w_test)
					log_str = "Most co-appear to %s: in collection %i: " % (test_word, i+1)
					for k in xrange(top_k):
						close_word = idx2word[closest[k]]
						log_str = "%s %s," % (log_str, close_word)	
					print(log_str)


	# GLobal plotting
	
	tsne = TSNE(perplexity=10, n_components=2, init='pca', n_iter=5000)
	pca = PCA(n_components=2)

	plot_only = vocab_size - 3
	low_dim_embs = tsne.fit_transform(normalized_embeddings.eval())
	labels = [idx2word[i] for i in xrange(plot_only)]
	plot_with_labels(low_dim_embs, labels, "tsne.png")

	low_dim_embs = pca.fit_transform(normalized_embeddings.eval())
	plot_with_labels(low_dim_embs, labels, "pca.png")

	# ----------------------------------------------------------------------------------------------------

	# Collection level plotting
	plot_only = vocab_size - 3
	if len(hidden_size) > 0: # Always though
		coded_size = hidden_size[len(hidden_size)-1]
	else: # Just for debugging
		coded_size = 2 * embedding_size


	# collection from 1 to 3 ... I don't want to start from 0 though ...
	for c in xrange(1, 4):
		coded_embeddings = np.zeros((plot_only, coded_size), dtype='float32')
		context_word = "c_" + str(c)
		
		for i in xrange(plot_only):
			test = [i, word2idx[context_word]]
			tensor = np.asarray(test, dtype=np.int32)

			coded_vector = get_embedding(tensor)
			coded_embeddings[i] = coded_vector

		labels = [idx2word[i] for i in xrange(plot_only)]
		
		low_dim_embs = tsne.fit_transform(coded_embeddings)
		plot_with_labels(low_dim_embs, labels, "tsne_" + str(c) + ".png")

		low_dim_embs = pca.fit_transform(coded_embeddings)
		plot_with_labels(low_dim_embs, labels, "pca_" + str(c) + ".png")



if __name__ == "__main__":

	main()