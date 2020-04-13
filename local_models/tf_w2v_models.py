import collections
import itertools
import numpy as np
import tensorflow as tf
import sklearn
import tensorflow_models
import logging
import local_models.loggin

logger = logging.getLogger(__name__)

#################### Util functions #################### 

class fuckit(object):
    def __init__(self, message=None):
        self.message = message
    def __enter__(self): return self
    def __exit__(self, *args):
        if self.message is not None:
            print(self.message)
        return True

def build_dataset(sentences, vocabulary_size=50000):
    '''
    Build the dictionary and replace rare words with UNK token.
    
    Parameters
    ----------
    sentences: list of list of tokens
    vocabulary_size: maximum number of top occurring tokens to produce, 
        rare tokens will be replaced by 'UNK'
    '''
    words = itertools.chain(*sentences)
    count = collections.Counter()
    total = 0
    for word in words:
        count[word] += 1
        total += 1
    vocab = dict(count.most_common(vocabulary_size-1))

    word_index_mapping = {word: i+1 for i, word in enumerate(vocab)} # {word: index}
    word_index_mapping["UNK"] = 0

    data = []
    k = 0
    for i, sentence in enumerate(sentences):
        data.append([])
        for j, word in enumerate(sentence):
            if word in word_index_mapping:
                data[-1].append(word_index_mapping[word])
            else:
                data[-1].append(word_index_mapping["UNK"])
                count["UNK"] += 1
    reverse_mapping = dict(zip(word_index_mapping.values(), word_index_mapping.keys()))
    return data, count, word_index_mapping, reverse_mapping

def build_dictionary(sentences, vocabulary_size=50000):
    '''
    Build the dictionary and replace rare words with UNK token.
    
    Parameters
    ----------
    sentences: list of list of tokens
    vocabulary_size: maximum number of top occurring tokens to produce, 
        rare tokens will be replaced by 'UNK'
    '''
    words = itertools.chain(*sentences)
    count = collections.Counter()
    total = 0
    for word in words:
        count[word] += 1
        total += 1
    vocab = dict(count.most_common(vocabulary_size-1))

    word_index_mapping = {word: i+1 for i, word in enumerate(vocab)} # {word: index}
    word_index_mapping["UNK"] = 0

    reverse_mapping = dict(zip(word_index_mapping.values(), word_index_mapping.keys()))
    return word_index_mapping, reverse_mapping

def build_dataset_predictionary(sentences, dictionary):
    '''
    Build the dictionary and replace rare words with UNK token.
    
    Parameters
    ----------
    sentences: list of list of tokens
    vocabulary_size: maximum number of top occurring tokens to produce, 
        rare tokens will be replaced by 'UNK'
    '''
    words = itertools.chain(*sentences)

    data = []
    k = 0
    for i, sentence in enumerate(sentences):
        data.append([])
        for j, word in enumerate(sentence):
            if word in dictionary:
                data[-1].append(dictionary[word])
            else:
                data[-1].append(dictionary["UNK"])
    return data

def generate_batch_skipgram(data, batch_size, num_skips, skip_window):
    '''
    Batch generator for skip-gram model.
    
    Parameters
    ----------
    data: list of list of words (or index of words)
    batch_size: number of words in each mini-batch
    num_skips: number of surrounding words on either direction (1: one word ahead and one word following)
    skip_window: number of words at both ends of a sentence to skip (1: skip the first and last word of a sentence)    
    '''
    assert num_skips <= skip_window
    sentence_index = 0
    data_index = 0
    word_index = skip_window
    target_index = -num_skips
    while True:
        batch = np.ndarray(shape=(batch_size,1,2), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size,2), dtype=np.int32)
        i = 0
        while i < batch_size:
            batch[i,0] = sentence_index, word_index
            labels[i] = sentence_index, word_index + target_index
            target_index += 1
            if word_index + target_index == min(word_index + num_skips + 1, len(data[sentence_index])): #end of num_skips window... move onto next center word
                target_index = -num_skips
                word_index += 1
                if word_index >= len(data[sentence_index]) - skip_window: #increment the sentence
                    sentence_index = (sentence_index + 1) % len(data)
                    word_index = skip_window
            else: #move to the next target, skipping 0
                target_index += 1 + (not target_index)

        yield batch, labels


def generate_batch_cbow(data, batch_size, num_skips, skip_window):
    '''
    Batch generator for CBOW (Continuous Bag of Words).
    batch should be a shape of (batch_size, num_skips)

    Parameters
    ----------
    data: list of list of words (or index of words)
    batch_size: number of words in each mini-batch
    num_skips: number of surrounding words in either direction (1: one word ahead and one word following)
    skip_window: number of words at both ends of a sentence to skip (1: skip the first and last word of a sentence)
    '''

    assert num_skips <= skip_window
    sentence_index = 0
    word_index = skip_window
    target_window = np.concatenate((np.arange(-num_skips, 0), np.arange(1, num_skips+1)))
    while True:
        batch = np.ndarray(shape=(batch_size,2*num_skips,2), dtype=np.int32) #batch, which_skiptarget, (sentence_index, word_index)
        labels = np.ndarray(shape=(batch_size,2), dtype=np.int32)
        for i in range(batch_size):
            batch[i,:,0] = sentence_index
            batch[i,:,1] = target_window + word_index
            labels[i] = sentence_index, word_index

            word_index += 1
            if word_index >= len(data[sentence_index]) - skip_window: #increment the sentence
                sentence_index = (sentence_index + 1) % len(data)
                while len(data[sentence_index]) < 2*num_skips + 1: #skip sentences that are too short
                    sentence_index = (sentence_index + 1) % len(data)
                word_index = skip_window

        yield batch, labels
        
def sentenceindex_2_word_list(sentences, word_indices):
    x = np.zeros(word_indices.shape[:-1])
    for x_indices in itertools.product(*map(range, x.shape)):
        try:
            s_i, w_i = word_indices[x_indices]
            x[x_indices] = sentences[s_i][w_i]
        except Exception as e:
            print(len(sentences), len(sentences[s_i]), s_i, w_i)
            raise e
    return x

class Word2Vec(sklearn.base.BaseEstimator):

    def __init__(self, batch_size=128, num_skips=2, skip_window=2, 
        architecture='cbow', embedding_size=128, vocabulary_size=5000, 
        loss_type='sampled_softmax_loss', n_neg_samples=64,
        optimizer="AdagradOptimizer", log_epochs=True,
        learning_rate=1.0, epochs=10000, 
        #fitted params for cloning and pickling
        dictionary=None, reverse_dictionary=None, final_embeddings=None, beta0=None, model=None,
        trainable_embeddings=True, trainable_weights=True, trainable_bias=True): 
        import tensorflow
        
        # bind params to class
        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window
        self.architecture = architecture
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.loss_type = loss_type
        self.n_neg_samples = n_neg_samples 
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.log_epochs=log_epochs
        self.model=model
        self.trainable_embeddings=trainable_embeddings
        self.trainable_weights=trainable_weights
        self.trainable_bias=trainable_bias
        
        self.dictionary=dictionary
        self.reverse_dictionary=reverse_dictionary
        self.final_embeddings=final_embeddings
        self.beta0 = beta0

        # choose a batch_generator function for feed_dict
        self._choose_batch_generator()
        
        super().__init__()

    def __setstate__(self, d):
        self.__dict__.update(d)
        #self.model = None
        self.build_model()
    
    def __getstate__(self):
        with fuckit(): self.sess.close()
        with fuckit(): del self.sess 
        with fuckit():
            with self.model.graph_.as_default():
                tf.reset_default_graph()
        with fuckit(): del self.model
        return self.__dict__    

    def init_vars_(self, beta0=None):

        if beta0 is not None:
            embeddings, weights, biases = (
                beta0[:self.vocabulary_size*self.embedding_size].reshape(
                    self.vocabulary_size, self.embedding_size),
                beta0[self.vocabulary_size*self.embedding_size:2*self.vocabulary_size*self.embedding_size].reshape(
                    self.vocabulary_size, self.embedding_size),
                beta0[2*self.vocabulary_size*self.embedding_size:])
            self.sess.run(
                (self.model.embeddings_feed,
                 self.model.weights_feed, 
                 self.model.biases_feed),
                feed_dict = {self.model.embeddings_ph: embeddings,
                    self.model.weights_ph: weights,
                    self.model.biases_ph: biases})

    def _choose_batch_generator(self):
        if self.architecture == 'skip-gram':
            self.generate_batch = generate_batch_skipgram
        elif self.architecture == 'cbow':
            self.generate_batch = generate_batch_cbow

    def build_model_(self):
        '''
        Init a tensorflow Graph containing:
        input data, variables, model, loss function, optimizer
        '''
        with tf.name_scope("w2v"):

            with tf.name_scope("input"):

                if self.architecture == 'skip-gram':
                    self.model.X = tf.placeholder(tf.int32, shape=[None,1])
                elif self.architecture == 'cbow':
                    self.model.X = tf.placeholder(tf.int32, shape=[None, 2*self.num_skips])
                self.model.y = tf.placeholder(tf.int32, shape=[None,1])
                self.model.sample_weight = tf.placeholder(tf.float32, shape=[None])
                
            with tf.name_scope("variables"):

                self.model.embeddings = tf.Variable(
                    tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0), name="embeddings")

                self.model.weights = tf.Variable(
                    tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                        stddev=1.0 / np.sqrt(self.embedding_size)), name="weights")

                self.model.biases = tf.Variable(tf.zeros([self.vocabulary_size]), name="biases")
                
                self.model.embeddings_ph = tf.placeholder(self.model.embeddings.dtype, shape=self.model.embeddings.shape)
                self.model.weights_ph = tf.placeholder(self.model.weights.dtype, shape=self.model.weights.shape)
                self.model.biases_ph = tf.placeholder(self.model.biases.dtype, shape=self.model.biases.shape)
                
                self.model.embeddings_feed = self.model.embeddings.assign(self.model.embeddings_ph)
                self.model.weights_feed = self.model.weights.assign(self.model.weights_ph)
                self.model.biases_feed = self.model.biases.assign(self.model.biases_ph)
  
            with tf.name_scope("embedding_layer"):
                embed = tf.nn.embedding_lookup(self.model.embeddings, self.model.X)
                self.model.embed = tf.reduce_sum(embed, axis=1)
            
            with tf.name_scope("loss"):
                if self.loss_type == "softmax_loss":
                    logits = tf.matmul(self.model.embed, tf.transpose(self.model.weights))
                    logits = tf.nn.bias_add(logits, self.model.biases)
                    labels_one_hot = tf.one_hot(self.model.y, self.vocabulary_size)
                    loss = tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels_one_hot,
                        logits=logits)
                if self.loss_type == 'sampled_softmax_loss':
                    loss = tf.nn.sampled_softmax_loss(self.model.weights, self.model.biases, self.model.y,
                        self.model.embed, self.n_neg_samples, self.vocabulary_size)
                elif self.loss_type == 'nce_loss':
                    loss = tf.nn.nce_loss(self.model.weights, self.model.biases, self.model.y, 
                        self.model.embed, self.n_neg_samples, self.vocabulary_size)
                self.model.loss = tf.reduce_mean(loss*self.model.sample_weight)

            with tf.name_scope("optimization"):
                self.model.optimizer = getattr(tf.train, self.optimizer)(self.learning_rate, name="optimizer")
                self.model.grads = self.model.optimizer.compute_gradients(self.model.loss)
                self.model.grad_application = self.model.optimizer.apply_gradients(self.model.grads)
                with tf.control_dependencies([self.model.grad_application]):
                    self.model.train_step = tf.no_op(name="train_step")
  
            # init op 
            self.model.initializer = tf.initialize_all_variables()
            
    def get_global_sess(self):
        global TFGlobalSesh
        try:
            TFGlobalSesh
        except:
            TFGlobalSesh = tf.Session(graph = self.model.graph_)
        self.sess = TFGlobalSesh
        return self.sess

    def get_global_graph(self):
        global TFGlobalGraph
        try:
            TFGlobalGraph
        except:
            TFGlobalGraph = tf.Graph()
        self.model.graph_ = TFGlobalGraph
        return self.model.graph_

    def build_model(self):
        ''' Initializes a new graph, and then calls the .build_model_ method, which must be implemented by a TFEstimator '''
        self.model = tensorflow_models.NameSpace()
        self.get_global_graph()
        with self.model.graph_.as_default():
            self.build_model_()

    def _build_dictionaries(self, sentences):
        '''
        Process tokens and build dictionaries mapping between tokens and 
        their indices. Also generate token count and bind these to self.
        '''

        data, count, dictionary, reverse_dictionary = build_dataset(sentences, 
            self.vocabulary_size)
        try:
            assert len(dictionary) >= self.vocabulary_size
        except Exception as e:
            print("number of available words must be >= self.vocabulary_size")
            raise e
        self.dictionary = dictionary
        self.reverse_dictionary = reverse_dictionary
        self.count = count
        return data


    def fit(self, X, y=None, sample_weight=None, beta0=None,**fit_params):
        ''' Puts inputs into np format, initializes a session, builds the graph and then calls `.fit_` 
                which can be overridden by individual models '''
            
        beta0 = beta0 if beta0 is not None else self.beta0
        if self.model is None:
            logger.info("building model!")
            self.build_model()
        if self.dictionary is None:
            self.dictionary, self.reverse_dictionary = build_dictionary(X, self.vocab_size)
            assert len(dictionary) == self.vocabulary_size
        data = build_dataset_predictionary(X, self.dictionary)

        with self.model.graph_.as_default():
            with tf.Session(graph = self.model.graph_) as self.sess:
                self.sess.run(self.model.initializer)
                self.init_vars_(beta0)
                fitted_model = self.fit_(data, sample_weight=sample_weight, **fit_params)
                embeddings, weights, biases = self.sess.run(
                    (self.model.embeddings, self.model.weights, self.model.biases))
                self.beta0 = np.concatenate([x.flatten() for x in
                    [embeddings, weights, biases]])
                self.embeddings = embeddings
                self.final_embeddings = embeddings / np.linalg.norm(embeddings)
        
        return fitted_model
        
    def fit_(self, X, sample_weight=None, feed_dict_extras={}):
        ''' Trains for a number of epochs.  Model input must be in self.model.X, output in self.model.y, loss in self.model.loss, and training using self.model.train_step '''
        # sample_weight is *PER SENTENCE*

        logger.info("fitting w2v model")
        epoch_log_str = "epoch: {:06d} ::: loss: {:.02e}"
    
        batch_generator = self.generate_batch(X, self.batch_size, self.num_skips, self.skip_window)
        if sample_weight is None:
            sample_weight = np.ones(len(X))
        for batch_epoch in range(int(self.epochs*len(X)/self.batch_size)):
            epoch = int(batch_epoch*self.batch_size/len(X))
            batch_indices, labels_indices = next(batch_generator)
            batch, labels = sentenceindex_2_word_list(X, batch_indices), sentenceindex_2_word_list(X, labels_indices)
            batch_sample_weight = sample_weight[labels_indices[:,0]]

            feed_dict={
                self.model.X: batch,
                self.model.y: labels.reshape(-1,1),
                self.model.sample_weight: batch_sample_weight
            }
            
            loss, _ = self.sess.run(
                (self.model.loss, self.model.train_step),
                feed_dict
            )

            if self.log_epochs and not (isinstance(self.log_epochs, float) and epoch%int(1/self.log_epochs)):
                logger.info(epoch_log_str.format(epoch, loss.sum()))

        logger.info("finished fitting :::: loss: " + str(loss.sum()))
        self.final_loss = loss.sum()

        return self


    def transform(self, words):
        '''
        Look up embedding vectors using indices
        words: list of words
        '''
        # make sure all word index are in range
        indices = [self.dictionary[w] for w in words]
        return self.final_embeddings[indices]


    def sort(self, word):
        '''
        Use an input word to sort words using cosine distance in ascending order
        '''
        i = self.dictionary[word]
        vec = self.final_embeddings[i].reshape(1, -1)
        # Calculate pairwise cosine distance and flatten to 1-d
        pdist = sklearn.metrics.pairwise_distances(self.final_embeddings, vec, metric='cosine').ravel()
        return [self.reverse_dictionary[j] for j in pdist.argsort()]
