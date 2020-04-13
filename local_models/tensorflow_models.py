import tensorflow as tf
import sklearn
import numpy as np
import os
import zipfile
import random
import uuid

TF_CONFIG_ = tf.ConfigProto()
TF_CONFIG_.gpu_options.allow_growth = True

SESSIONS_ = []
def n_opened_sessions():
    ''' Stuff in here keeps a reference to every opened session in SESSIONS_.
        Useful for debugging rogue tf sessions lying about '''
    return str((len(SESSIONS_) - sum(list(map(lambda x: x._closed, SESSIONS_))))) + "/" + str(len(SESSIONS_))

def np_batcher(n, batch_size=None):
    ''' Batch data from a numpy array '''
    if batch_size is None:
        yield slice(None)
        return
    i = 0
    while i < n:
        yield slice(i, i + batch_size)
        i += batch_size
    
class NameSpace(object):
    ''' This is to create a dict-like object that allows access via dot syntax.
        e.g. `stuff_keeper = NameSpace(); stuff_keeper.some_stuff = some_stuff_to_keep` '''
    def __deepcopy__(self, *args):
        return self
    def __copy__(self, *args):
        return self

class PickleableTFModel(object):
    ''' This mixin allows a tensorflow model class to be pickleable.
    
        Your tensorflow model must implement a `build_model` method, and store all of the 
        tf.Variables (and probably the rest of in an attribute `self.model` as attributes of `self.model` 
        
        Before saving, your tensorflow model must be initialized into a session stored in `self.sess`'''
    TEMP_MODEL_FILE_ = "tmp_tf_model"
    
    def get_global_sess(self):
        global TFGlobalSesh
        try:
            TFGlobalSesh
        except:
            TFGlobalSesh = tf.Session(graph = self.model.graph_)
        self.sess = TFGlobalSesh

    def get_global_graph(self):
        global TFGlobalGraph
        try:
            TFGlobalGraph
        except:
            TFGlobalGraph = tf.Graph()
        self.model.graph_ = TFGlobalGraph

    def __getvariables__(self):
        ''' helper method that goes through the `self.model` and picks out which things are `tf.Variables` '''
        with self.model.graph_.as_default():
            d = dict(self.__dict__)
            variables = {}
            if hasattr(self, "model"):
                for attr_name, attr in dict(d["model"].__dict__).items():
                    if isinstance(attr, tf.Variable):
                        variables[attr_name] = attr
            return variables
    
    def __getstate__(self):
        log.info("pickling tf model")
        with self.model.graph_.as_default():
            d = dict(self.__dict__)
            if hasattr(self, "model"):
                #get the sess variables
                try:
                    sess_saver = tf.train.Saver(self.__getvariables__())
                    #sess_saver = tf.train.Saver()
                    sess_filename = PickleableTFModel.TEMP_MODEL_FILE_ + str(uuid.uuid4())
                    path = os.path.join(os.getcwd(), sess_filename)
                    sess_saver.save(self.sess, path)
                    files = [f for f in os.listdir(os.getcwd()) if sess_filename in f]
                    log.info(str(files))
                    with zipfile.ZipFile(path, mode='w') as zf:
                        for f in files:
                            zf.write(f)
                    with open(path, "rb") as f:
                        serial_model_data = f.read()
                    d["model"] = serial_model_data
                    del d["sess"]
                finally:
                    os.remove(path)
                    for f in files:
                        os.remove(f)
            return d
    
    def __setstate__(self, d):
        log.info("unpickling tf model")
        if "model" in d:
            model = d["model"]
            del d["model"]
            self.__dict__.update(d)
            name = PickleableTFModel.TEMP_MODEL_FILE_ + str(random.random())
            path = os.path.join(os.getcwd(), name)
            try:
                log.info("writing model bytes")
                with open(path, "wb") as f:
                    f.write(model)
                log.info("extracting zip file")
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall()
                ckpt = ".".join(zf.namelist()[0].split(".")[:-1])
                log.info("building model")
                self.build_model()
                with self.model.graph_.as_default():
                    saver = tf.train.Saver(self.__getvariables__())
                    log.info("creating session")
                    self.sess = tf.Session(config = TF_CONFIG_, graph = self.model.graph_)
                    #self.get_global_sess()
                    log.info("restoring checkpoint: " + ckpt)
                    saver.restore(self.sess, os.path.join(".", ckpt))
                log.info("tf model restored!")
            except Exception as e:
                log.exception(e)
            finally:
                log.info("destroying the evidence")
                files = zf.namelist()
                os.remove(path)
                for f in files:
                    os.remove(f)
        else:
            self.__dict__.update(d)
  

class TFEstimatorMixin(object):
    ''' This class implements helpful things that our tensorflow models will generally want to have,
        such as initializing the graph, initializing the session, building the model, setting input and output shapes
        for use in the model, and iterating a training step over a number of epochs '''
    def build_model(self):
        ''' Initializes a new graph, and then calls the .build_model_ method, which must be implemented by a TFEstimator '''
        self.model = NameSpace()
        self.model.graph_ = tf.Graph()
        with self.model.graph_.as_default():
            self.build_model_()
    def build_model_(self):
        raise NotImplementedError()
            
    def fit(self, X, y=None, sample_weight=None, **fit_params):
        ''' Puts inputs into np format, initializes a session, builds the graph and then calls `.fit_` which can be overridden by individual models '''
        if sample_weight is None: 
            sample_weight = np.ones(X.shape[0])
        else:
            sample_weight = np.array(sample_weight)
        self.input_shape_ = X.shape[1:]
        self.set_output_shape_(y)
        self.build_model()
        with self.model.graph_.as_default():
            self.sess = tf.Session(config = TF_CONFIG_, graph = self.model.graph_)
            SESSIONS_.append(self.sess)
            initializer = tf.global_variables_initializer()
            self.sess.run(initializer)
            return self.fit_(X, y, sample_weight=sample_weight, **fit_params)
    def set_output_shape_(*args, **kwargs):
        ''' Must implement this for the fit method to work '''
        raise NotImplementedError()
       
    def fit_(self, X, y=None, sample_weight=None, feed_dict_extras={}):
        ''' Trains for a number of epochs.  Model input must be in self.model.x, output in self.model.y, loss in self.model.loss, and training using self.model.train_step '''
        if not hasattr(self, "batch_size"):
            self.batch_size = None
        
        for epoch in range(self.n_epochs):
            batcher = np_batcher(X.shape[0], self.batch_size)
            for batch in batcher:
                feed_dict={
                    self.model.x : X[batch],
                }
                if y is not None:
                    feed_dict[self.model.y] = y[batch]
                if sample_weight is not None:
                    feed_dict[self.model.sample_weight] = sample_weight[batch]
                feed_dict.update(feed_dict_extras)
                loss, _ = self.sess.run(
                    (self.model.loss, self.model.train_step),
                    feed_dict
                )

                log.info("epoch: " + str(epoch) + " :::: loss: " + str(loss.sum()))

        return self
  
  
class TFClassifierMixin(TFEstimatorMixin):
    ''' Has specific things that would be useful for classifiers.'''

    def set_output_shape_(self, y):
        ''' This sets the number of classes and the vectorized output shape for a categorical variable '''
        self.classes_ = np.unique(y)
        self.output_shape_ = (len(self.classes_),)
    
    def predict_proba(self, X):
        ''' Assuming that self.model implements a predict_proba tensorflow graph element '''
        return self.sess.run(
            self.model.predict_proba, 
            feed_dict={
                self.model.x : X
            })
            
    def fit_(self, X, y, sample_weight=None, feed_dict_extras={}):
        ''' This does some additional transformation on y before eventually fitting the model '''
        label_encoder = sklearn.preprocessing.LabelEncoder()
        label_encoder.classes_ = self.classes_
        onehot_encoder = sklearn.preprocessing.OneHotEncoder().fit(np.arange(len(label_encoder.classes_)).reshape((-1,1)))
        y = onehot_encoder.transform(label_encoder.transform(y).reshape((-1,1))).toarray()
        return super().fit_(X,y,sample_weight=sample_weight)
            
    def predict(self,X):
        ''' Returns the most likely class '''
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]     

        
class TFRegressorMixin(TFEstimatorMixin):
    ''' Has specific things that would be useful for regressors.'''

    def set_output_shape_(self, y):
        ''' logic to handle 1d y variables '''
        if len(y.shape) > 1:
            self.output_shape_ = y.shape[1:]
        else:
            self.output_shape_ = (1,) 
               
    def fit_(self, X, y, sample_weight=None, feed_dict_extras={}):
        ''' more logic to handle 1d y variables prior to fitting '''
        if (y is not None) and (len(y.shape) == 1):
            y = y.reshape((-1,1))
        return super().fit_(X,y,sample_weight=sample_weight, feed_dict_extras=feed_dict_extras)
               
    def predict(self,X):
        ''' Assumes that self.model.predictions is a tf graph element returning a prediction ''' 
        return self.sess.run(
            self.model.predictions,
            feed_dict={
                self.model.x : X
            })
            
        
class OneLayerNNClassifier(sklearn.base.BaseEstimator, PickleableTFModel, TFClassifierMixin):
    ''' A one hidden layer NN Classifier.  Hidden layer is activated by a relu.
        Parameters:
            n_hidden: number of hidden neurons
            n_epochs: number of training epochs
            learning_rate: make it bigger to learn faster, at the risk of killing your relu
            trainable: set to false if you want to not allow training.  For example, if you want to use this as part of another network
            regularization: penalize large weights (higher is more penalization)
            dropout: the default dropout rate on the hidden layer (note that 1 == NO DROPOUT, 0 == 100% DROPOUT)
            batch_size: defaults to the entire dataset.
    '''
    def __init__(self, n_hidden=20, n_epochs=1000, learning_rate=0.01, trainable=True, regularization=0.01, dropout=1.0, batch_size=None):
        self.n_hidden = n_hidden
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.trainable = trainable
        self.regularization = regularization
        self.dropout = dropout
        self.batch_size = batch_size
        super().__init__()
    
    def fit_(self, X, y, sample_weight=None, feed_dict_extras={}):
        ''' Feeds the extra feed_dict thing `keep_prob` to the network, to allow adjusting the dropout '''
        feed_dict = {self.model.keep_prob: self.dropout}
        feed_dict.update(feed_dict_extras)
        return super().fit_(X,y,sample_weight=sample_weight, feed_dict_extras=feed_dict)
    
    def build_model_(self):
        ''' The actual network architecture '''
        with tf.name_scope("one_hidden"):
        
            with tf.name_scope("layer_0_input"):
                input_dim = [None] + list(self.input_shape_)
                layer_0_output_dim = list(self.input_shape_)
                output_dim = [None] + list(self.output_shape_)
                self.model.x = tf.placeholder(tf.float32, shape=input_dim, name="input")
                self.model.y = tf.placeholder(tf.float32, shape=output_dim, name="expected_output")
                self.model.sample_weight = tf.placeholder(tf.float32, shape=[None], name="sample_weight")
                
            with tf.name_scope("fc1"):
                layer_1_input_dim = layer_0_output_dim
                layer_1_output_dim = [integerify(self.n_hidden)]
                self.model.b1 = tf.Variable(tf.constant(0.1, shape=layer_1_output_dim), name="bias", trainable=self.trainable)
                self.model.w1 = tf.Variable(tf.truncated_normal(layer_1_input_dim + layer_1_output_dim, stddev=0.1), name="weights", trainable=self.trainable)
                self.model.fc1 = tf.nn.relu(tf.matmul(self.model.x, self.model.w1) + self.model.b1, name="relu")
                
                self.model.keep_prob = tf.placeholder_with_default(1.0, shape=())
                self.model.dropout1 = tf.nn.dropout(self.model.fc1, self.model.keep_prob, name="dropout")

                self.model.weight_decay_layer_1 = tf.Variable(self.regularization, trainable=False, name="weight_decay_rate")
                self.model.layer_1_loss = tf.multiply(tf.nn.l2_loss(self.model.w1), self.model.weight_decay_layer_1, name="weight_loss")
                
            with tf.name_scope("fc2"):
                layer_2_input_dim = layer_1_output_dim
                layer_2_output_dim = [len(self.classes_)]
                self.model.b2 = tf.Variable(tf.constant(0.1, shape=layer_2_output_dim), name="bias", trainable=self.trainable)
                self.model.w2 = tf.Variable(tf.truncated_normal(layer_2_input_dim + layer_2_output_dim, stddev=0.1), name="weights", trainable=self.trainable)
                self.model.fc2 = tf.matmul(self.model.dropout1, self.model.w2) + self.model.b2

                self.model.weight_decay_layer_2 = tf.Variable(self.regularization, trainable=False, name="weight_decay_rate")
                self.model.layer_2_loss = tf.multiply(tf.nn.l2_loss(self.model.w2), self.model.weight_decay_layer_2, name="weight_loss")

            with tf.name_scope("output"):
                self.model.logits = self.model.fc2
                self.model.predict_proba = tf.nn.softmax(self.model.logits, name="proba")
                self.model.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.model.y, logits=self.model.logits, name="cross_entropy_loss")
                self.model.loss = tf.add_n((tf.reduce_mean(tf.multiply(self.model.cross_entropy_loss, self.model.sample_weight)), self.model.layer_1_loss, self.model.layer_2_loss))
                
            if self.trainable:
                with tf.name_scope("training"): 
                    self.model.optimizer = tf.train.AdamOptimizer(self.learning_rate, name="optimizer")
                    self.model.grads = self.model.optimizer.compute_gradients(self.model.loss)
                    self.model.grad_application = self.model.optimizer.apply_gradients(self.model.grads)
                    with tf.control_dependencies([self.model.grad_application]):
                        self.model.train_step = tf.no_op(name="train_step")

        return self.model
        
class LogisticRegression(sklearn.base.BaseEstimator, PickleableTFModel, TFClassifierMixin):
    ''' A zero hidden layer NN Classifier (i.e. Logit Regression).
        Parameters:
            n_epochs: number of training epochs
            learning_rate: make it bigger to learn faster, at the risk of killing your relu
            trainable: set to false if you want to not allow training.  For example, if you want to use this as part of another network
            regularization: penalize large weights (higher is more penalization)
            dropout: the default dropout rate on the hidden layer (note that 1 == NO DROPOUT, 0 == 100% DROPOUT)
            batch_size: defaults to the entire dataset.
    '''
    def __init__(self, n_epochs=1000, learning_rate=0.01, trainable=True, regularization=0.01, batch_size=None):
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.trainable = trainable
        self.regularization = regularization
        self.batch_size = batch_size
        super().__init__()
    
    def build_model_(self):
        ''' The actual network architecture '''
        with tf.name_scope("logistic_regression"):
        
            with tf.name_scope("layer_0_input"):
                input_dim = [None] + list(self.input_shape_)
                layer_0_output_dim = list(self.input_shape_)
                output_dim = [None] + list(self.output_shape_)    
                self.model.x = tf.placeholder(tf.float32, shape=input_dim, name="input")
                self.model.y = tf.placeholder(tf.float32, shape=output_dim, name="expected_output")
                self.model.sample_weight = tf.placeholder(tf.float32, shape=[None], name="sample_weight")
                
            with tf.name_scope("fc1"):
                layer_1_input_dim = layer_0_output_dim
                layer_1_output_dim = [len(self.classes_)]
                self.model.b1 = tf.Variable(tf.constant(0.1, shape=layer_1_output_dim), name="bias", trainable=self.trainable)
                self.model.w1 = tf.Variable(tf.truncated_normal(layer_1_input_dim + layer_1_output_dim, stddev=0.1), name="weights", trainable=self.trainable)
                self.model.fc1 = tf.nn.relu(tf.matmul(self.model.x, self.model.w1) + self.model.b1, name="relu")
                
                self.model.weight_decay_layer_1 = tf.Variable(self.regularization, trainable=False, name="weight_decay_rate")
                self.model.layer_1_loss = tf.multiply(tf.nn.l2_loss(self.model.w1), self.model.weight_decay_layer_1, name="weight_loss")

            with tf.name_scope("output"):
                self.model.logits = self.model.fc1
                self.model.predict_proba = tf.nn.softmax(self.model.logits, name="proba")
                self.model.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.model.y, logits=self.model.logits, name="cross_entropy_loss")
                self.model.loss = tf.add_n((tf.reduce_mean(tf.multiply(self.model.cross_entropy_loss, self.model.sample_weight)), self.model.layer_1_loss))
                
            if self.trainable:
                with tf.name_scope("training"): 
                    self.model.optimizer = tf.train.AdamOptimizer(self.learning_rate, name="optimizer")
                    self.model.grads = self.model.optimizer.compute_gradients(self.model.loss)
                    self.model.grad_application = self.model.optimizer.apply_gradients(self.model.grads)
                    with tf.control_dependencies([self.model.grad_application]):
                        self.model.train_step = tf.no_op(name="train_step")

        return self.model
        
class OneLayerNNRegressor(sklearn.base.BaseEstimator, PickleableTFModel, TFRegressorMixin):
    ''' A one hidden layer NN Classifier.  Hidden layer is activated by a relu.
        Parameters:
            n_hidden: number of hidden neurons
            n_epochs: number of training epochs
            learning_rate: make it bigger to learn faster, at the risk of killing your relu
            trainable: set to false if you want to not allow training.  For example, if you want to use this as part of another network
            regularization: penalize large weights (higher is more penalization)
            dropout: the default dropout rate on the hidden layer (note that 1 == NO DROPOUT, 0 == 100% DROPOUT)
            batch_size: defaults to the entire dataset.
    '''
    def __init__(self, n_hidden=20, n_epochs=1000, learning_rate=0.01, trainable=True, regularization=0.01, dropout=0.5, batch_size=None, scale=1.0):
        self.n_hidden = n_hidden
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.trainable = trainable
        self.regularization = regularization
        self.dropout = dropout
        self.batch_size = batch_size
        self.scale = scale
        super().__init__()
    
    def fit_(self, X, y, sample_weight=None, feed_dict_extras={}):
        ''' Feeds the extra feed_dict thing `keep_prob` to the network, to allow adjusting the dropout '''
        feed_dict = {self.model.keep_prob: self.dropout}
        feed_dict.update(feed_dict_extras)
        return super().fit_(X,y,sample_weight=sample_weight, feed_dict_extras=feed_dict)
    
    def build_model_(self):
        ''' The actual network architecture '''
        with tf.name_scope("one_hidden"):

            with tf.name_scope("layer_0_input"):
                input_dim = [None] + list(self.input_shape_)
                layer_0_output_dim = list(self.input_shape_)
                output_dim = [None] + list(self.output_shape_)
                
                self.model.x = tf.placeholder(tf.float32, shape=input_dim, name="input")
                self.model.y = tf.placeholder(tf.float32, shape=output_dim, name="expected_output")
                batch_size = tf.shape(self.model.x)[0]
                self.model.sample_weight = tf.placeholder(tf.float32, shape=[None], name="sample_weight")
                sample_weight = tf.reshape(self.model.sample_weight, [-1,1])
                
            with tf.name_scope("fc1"):
                layer_1_input_dim = layer_0_output_dim
                layer_1_output_dim = [integerify(self.n_hidden)]
                self.model.b1 = tf.Variable(tf.constant(self.scale, shape=layer_1_output_dim), name="bias", trainable=self.trainable)
                self.model.w1 = tf.Variable(tf.truncated_normal(layer_1_input_dim + layer_1_output_dim, mean=2*self.scale, stddev=self.scale), name="weights", trainable=self.trainable)
                self.model.fc1 = tf.nn.relu(tf.matmul(self.model.x, self.model.w1) + self.model.b1, name="relu")
                
                self.model.keep_prob = tf.placeholder_with_default(1.0, shape=())
                self.model.dropout1 = tf.nn.dropout(self.model.fc1, self.model.keep_prob, name="dropout")

                self.model.weight_decay_layer_1 = tf.Variable(self.regularization, trainable=False, name="weight_decay_rate")
                self.model.layer_1_loss = tf.multiply(tf.nn.l2_loss(self.model.w1), self.model.weight_decay_layer_1, name="weight_loss")
                
            with tf.name_scope("fc2"):
                layer_2_input_dim = layer_1_output_dim
                layer_2_output_dim = list(self.output_shape_)
                self.model.b2 = tf.Variable(tf.constant(self.scale, shape=layer_2_output_dim), name="bias", trainable=self.trainable)
                self.model.w2 = tf.Variable(tf.truncated_normal(layer_2_input_dim + layer_2_output_dim, mean=2*self.scale, stddev=self.scale), name="weights", trainable=self.trainable)
                self.model.fc2 = tf.matmul(self.model.dropout1, self.model.w2) + self.model.b2

                self.model.weight_decay_layer_2 = tf.Variable(self.regularization, trainable=False, name="weight_decay_rate")
                self.model.layer_2_loss = tf.multiply(tf.nn.l2_loss(self.model.w2), self.model.weight_decay_layer_2, name="weight_loss")

            with tf.name_scope("output"):
                self.model.predictions = self.model.fc2
                self.model.rmse_loss = tf.losses.mean_squared_error(self.model.y, self.model.predictions, weights=sample_weight)
                self.model.loss = tf.add_n((self.model.rmse_loss, self.model.layer_1_loss, self.model.layer_2_loss))
                
            if self.trainable:
                with tf.name_scope("training"): 
                    self.model.optimizer = tf.train.AdamOptimizer(self.learning_rate, name="optimizer")
                    self.model.grads = self.model.optimizer.compute_gradients(self.model.loss)
                    self.model.grad_application = self.model.optimizer.apply_gradients(self.model.grads)
                    with tf.control_dependencies([self.model.grad_application]):
                        self.model.train_step = tf.no_op(name="train_step")

        return self.model
        
