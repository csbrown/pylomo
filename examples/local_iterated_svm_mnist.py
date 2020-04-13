import local_models.local_models
import numpy as np
import sklearn.linear_model
import sklearn.svm
import sklearn.preprocessing
import sklearn.datasets
import time
import os
import functools
import collections
import joblib
import local_models.utils
import pickle

np.random.seed(1)

mnist = sklearn.datasets.fetch_openml('mnist_784')
RUN = 3
#project_dir = "/home/usaswb/data/local_svm_mnist_{:02d}".format(RUN)
project_dir = "/home/guest/data/local_svm_mnist_{:02d}".format(RUN)
os.makedirs(project_dir, exist_ok=1)
models_dir = os.path.join(project_dir, "ovr_models")
os.makedirs(models_dir, exist_ok=1)

n_samples = mnist.data.shape[0]

SEED=1
train_data, test_data, train_labels, test_labels = sklearn.model_selection.train_test_split(
    mnist.data, mnist.target, test_size=0.3, random_state=SEED)

model = sklearn.svm.LinearSVC(C=3000., dual=False)
linear_models = local_models.local_models.LocalModels(model)
linear_models.fit(train_data, train_labels)

dz, iz = linear_models.index.query(train_data, k=2)
avg_1nn_dist = np.average(dz[:,1])
kernel = local_models.local_models.TriCubeKernel(bandwidth=avg_1nn_dist*20)

mnist_onehotifier = sklearn.preprocessing.OneHotEncoder([list(range(10))])
mnist_onehotifier.fit(np.array([list(range(10))]).T)
onehot_labels = mnist_onehotifier.transform(np.expand_dims(train_labels, 1))
onehot_labels = onehot_labels.toarray()

ovr_local_models = []
for i in range(onehot_labels.shape[1]):
    ovr_local_models.append(local_models.local_models.LocalModels(model))
    ovr_local_models[-1].fit(train_data, onehot_labels[:,i])    

train_predictions = []
for i in range(onehot_labels.shape[1]):
    iterator = orthogonal_project_scms(train_data, ovr_local_models[i], kernel, return_everything=False, alpha=0.3, parallel=True, 
        sharedmem=False, scms_iters=30, newt_iters=10, n_jobs=128)
    for j, (X,y,normals) in enumerate(iterator):
        np.savetxt(os.path.join(models_dir, "ovr_model_predictions_X_{:03d}_{:04d}".format(i,j)), X)
        np.savetxt(os.path.join(models_dir, "ovr_model_predictions_y_{:03d}_{:04d}".format(i,j)), y)
        np.savetxt(os.path.join(models_dir, "ovr_model_predictions_normals_{:03d}_{:04d}".format(i,j)), normals[:,0,:])

'''
train_predictions = []
for i in range(onehot_labels.shape[1]):
    X = np.loadtxt(os.path.join(models_dir, "ovr_model_predictions_X_{:03d}".format(i)))
    y = np.loadtxt(os.path.join(models_dir, "ovr_model_predictions_y_{:03d}".format(i)))
    normals = np.loadtxt(os.path.join(models_dir, "ovr_model_predictions_normals_{:03d}".format(i)))
    train_predictions.append((X,y,normals))
    print(X.shape, y.shape, normals.shape)


# In[44]:


train_predictions[0][1]


# In[45]:


train_pred_labels = []
train_pred_scores = []


# In[46]:


ovr_local_models[8].model_features


# In[47]:


np.unique(ovr_local_models[8].model_targets)


# In[50]:


start = time.time()
preds = []
scores = []
for i in range(onehot_labels.shape[1]):
    logger.info("training_scores {} {}".format(i, (time.time() - start)/3600))
    preds.append(ovr_local_models[i].predict(train_data, kernel=kernel, r=kernel.support_radius()))
    scores.append(np.sqrt(np.sum((preds[i][0] - preds[i][1])**2, axis=-1)))


# In[51]:


preds, scores


# In[ ]:





# In[52]:


test_predictions = []
for i in range(onehot_labels.shape[1]):
    test_predictions.append(
        orthogonal_project_scms(test_data, ovr_local_models[i], kernel, return_everything=False, alpha=0.3, parallel=True, sharedmem=False)
    )
    X,y,normals = test_predictions[-1]
    print(X.shape, y.shape, normals.shape)
    logger.info("fitted mnist test ovr model {}".format(i))
    np.savetxt(os.path.join(models_dir, "ovr_model_test_predictions_X_{:03d}".format(i)), X)
    np.savetxt(os.path.join(models_dir, "ovr_model_test_predictions_y_{:03d}".format(i)), y)
    np.savetxt(os.path.join(models_dir, "ovr_model_test_predictions_normals_{:03d}".format(i)), normals[:,0,:])


# In[54]:


np.sum(onehot_labels, axis=0)


# In[55]:


onehot_labels


# In[57]:


len(train_predictions), len(test_predictions)


# In[59]:


test_Xy = (test_predictions[0][0] - test_predictions[0][1])
test_Xy_normalized = test_Xy/np.sqrt(np.sum(test_Xy**2, axis=1,keepdims=True))


# In[69]:


train_Xy = (train_predictions[0][0] - train_predictions[0][1])


# In[67]:


np.abs(np.sum(test_predictions[0][2][:,0,:]*test_Xy_normalized, axis=1))


# In[ ]:





# In[103]:


platt_regularizers = []
for i, ovr_local_model in enumerate(ovr_local_models):
    print(i)
    platt_regularizers.append(sklearn.linear_model.LogisticRegression(C=1e-2))
    train_Xy = train_predictions[i][0] - train_predictions[i][1]
    train_Xy_len = np.sqrt(np.sum(train_Xy**2, axis=1))
    train_pred = ovr_local_model.predict(train_data, kernel=kernel, r=kernel.support_radius())
    regularizer_input = train_Xy_len*(train_pred[:,0] * 2 - 1)
    platt_regularizers[-1].fit(regularizer_input.reshape(-1,1), onehot_labels[:,i])


# In[137]:


plt.scatter(regularizer_input, onehot_labels[:,i])
grid = np.linspace(np.min(regularizer_input), np.max(regularizer_input), 100)
plt.plot(grid, platt_regularizers[-1].predict_proba(grid.reshape(-1,1))[:,1], c='r')
plt.xlabel("orthogonal distance to decision surface", size=22)
plt.ylabel("$P$", size=22)
plt.savefig(


# In[109]:


onehot_labels_test = mnist_onehotifier.transform(np.expand_dims(test_labels, 1)).toarray()
onehot_labels_test


# In[ ]:





# In[105]:


test_Xy = test_predictions[i][0] - test_predictions[i][1]
test_Xy_len = np.sqrt(np.sum(test_Xy**2, axis=1))
test_pred = ovr_local_model.predict(test_data, kernel=kernel, r=kernel.support_radius())
regularizer_input_test = test_Xy_len*(test_pred[:,0] * 2 - 1)


# In[110]:


plt.scatter(regularizer_input_test, onehot_labels_test[:,i])
grid = np.linspace(np.min(regularizer_input_test), np.max(regularizer_input_test), 100)
plt.plot(grid, platt_regularizers[-1].predict_proba(grid.reshape(-1,1))[:,1], c='r')


# In[114]:


regularized_output = []
for i, ovr_local_model in enumerate(ovr_local_models):
    print(i)
    test_Xy = test_predictions[i][0] - test_predictions[i][1]
    test_Xy_len = np.sqrt(np.sum(test_Xy**2, axis=1))
    test_pred = ovr_local_model.predict(test_data, kernel=kernel, r=kernel.support_radius())
    regularizer_input_test = test_Xy_len*(test_pred[:,0] * 2 - 1)
    regularized_output.append(platt_regularizers[i].predict_proba(regularizer_input_test.reshape(-1,1))[:,1])


# In[115]:


regularized_output = np.stack(regularized_output, axis=-1)


# In[116]:


regularized_output.shape


# In[117]:


test_data.shape


# In[118]:


regularized_output[0]


# In[121]:


hard_predictions_test = np.argmax(regularized_output, axis=-1)


# In[124]:


import sklearn.metrics
confusion_test = sklearn.metrics.confusion_matrix(test_labels, hard_predictions_test)
confusion_test


# In[130]:


accuracy, precision, recall = (
    sklearn.metrics.accuracy_score(test_labels, hard_predictions_test),
    sklearn.metrics.precision_score(test_labels, hard_predictions_test, average="macro"), 
    sklearn.metrics.recall_score(test_labels, hard_predictions_test, average="macro"))
accuracy, precision, recall


# In[ ]:





# In[ ]:





# In[ ]:





# In[138]:


def nonorthopred_transformator(m,q,x,y,w):
    scores = m.decision_function(x)
    preds = m.predict(x)
    regularizer_features = scores*(preds*2-1)
    regularizer = sklearn.linear_model.LogisticRegression(C=1e-2, fit_intercept=False)
    regularizer.fit(regularizer_features.reshape(-1,1), y, sample_weight=w)
    return regularizer.predict_proba(m.decision_function(q.reshape(1,-1)).reshape(-1,1))


# In[145]:


nonortho_proba_transformations = []
for i, ovr_local_model in enumerate(ovr_local_models):
    nonortho_proba_transformations.append(
        ovr_local_model.transform(test_data, kernel=kernel, r=kernel.support_radius(), weighted=True,
            model_postprocessor=nonorthopred_transformator))


# In[147]:


nonortho_proba_transformations[0].shape


# In[148]:


nonortho_proba_transformations = np.stack(nonortho_proba_transformations, axis=-1)


# In[149]:


nonortho_proba_transformations.shape


# In[152]:


nonortho_hard_predictions = np.argmax(nonortho_proba_transformations[:,0], axis=-1)


# In[153]:


nonortho_confusion = sklearn.metrics.confusion_matrix(test_labels, nonortho_hard_predictions)
nonortho_acc, nonortho_prec, nonortho_rec = (
    sklearn.metrics.accuracy_score(test_labels, nonortho_hard_predictions),
    sklearn.metrics.precision_score(test_labels, nonortho_hard_predictions, average="macro"), 
    sklearn.metrics.recall_score(test_labels, nonortho_hard_predictions, average="macro"))
nonortho_confusion, nonortho_acc, nonortho_prec, nonortho_rec


# In[ ]:



'''
