# Python Local Modeling Library

A Library to aid the construction of local models, making predictions from them and extracting features from them.
Accepts models in the scikit-learn style interface.
Has convenience methods for extracting parameters from some commonly encountered model families.

## Getting Started

### Prerequisites

Requires `numpy` and `sklearn` at a bare minimum.  Currently "requires" scipy, but that could probably be relaxed in a future refactor.

### Installation

``` 
$ git clone <this_repo> <your_repo_dir>
$ python3 -m venv <your_venv_dir>
$ source <your_venv_dir>/bin/activate
$ pip3 install <your_repo_dir>
```

### Basic Usage

Here is a basic example implementing a [LOESS][1] model:

```
from local_models.local_models import *
import numpy as np
import matplotlib.pyplot as plt

X_train = np.linspace(0,6,100).reshape(-1,1)
y_train = np.sin(X_train) + np.random.normal(loc=0,scale=0.3,size=X_train.shape)
y_train = y_train.flatten()
X_test = np.linspace(-1,7,1000).reshape(-1,1)

kernel = GaussianKernel(bandwidth = 1.)
LOESS = LocalModels(sklearn.linear_model.LinearRegression(), kernel=kernel)
LOESS.fit(X_train,y_train) # This just builds an index and stores x and y

y_pred = LOESS.predict(X_test) # This makes local predictions at these various points
model_features = LOESS.transform(X_test) 
# model_features is a (X_test.shape[0], X_test.shape[1] + 1) shaped array, containing the coefficients of the
# various independent variables and also the intercept of the individual local models

plt.plot(X_test, y_pred)
plt.plot(X_test, model_features)
plt.scatter(X_train, y_train,c='r')
plt.legend(["predictions", "slope", "intercept", "data"])
plt.show()
```

![](https://i.stack.imgur.com/dfXQV.png)

Note that the `transform` method is relatively simple in this example because there are defaults for this model family defined in `local_models.default_model_features`.  
If you use an arbitrary model family (besides `LinearRegression` and a few others), you will need to pass in a parameter `model_postprocessor` to the `transform` method.
`model_postprocessor` is a function with the signature `(locally_trained_model, query_point, x_train, y_train, weights)` that outputs a 1d numpy array of features.  
One obvious option is to extract the parameters of `locally_trained_model` (which are often stored as model-specific-named parameters of the trained model).
For the LOESS model above, this would look like:

``` 
def LOESS_postprocessor(trained_linear_model, *args):
    return np.concatenate((trained_linear_model.coef_, trained_linear_model.intercept_.reshape(-1)))

model_features = LOESS.transform(X_test, model_postprocessor = LOESS_postprocessor) 
```

Experiments from our forthcoming IJCNN paper on localized classifiers can be found in [the included examples](examples/local_iterated_svm_moons.ipynb).
Experiments from our paper on localized Gaussian Processes can be found [here](https://github.com/csbrown/pylomo/blob/master/examples/local_gpr_contrived_period.ipynb), [here](examples/Activity_recognition_gpr.ipynb) and [here](examples/Todd_eeg_local_gpr_all.ipynb).  Unfortunately, the data for the EEG analyses are unavailable to share.

## Authors

**CScott Brown**

Please cite one of the following if you use this library for scholarly research:

\[1\] Brown, CScott, and Ryan G. Benton. "Local Gaussian Process Features for Clinical Sensor Time Series." 2019 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 2019.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details





[1]: https://en.wikipedia.org/wiki/Local_regression 
