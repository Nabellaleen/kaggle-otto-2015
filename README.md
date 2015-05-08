# kaggle-otto-2015
Code for Otto Kaggle Challenge : https://www.kaggle.com/c/otto-group-product-classification-challenge

## Install

Install requirements using pip :

    pip install -r requirements.txt
    
Requirements contains libraries like numpy, asking to be compiled on your system. If you can't compile its or don't know how do process, visit their official websites to find binaries with installers.
You also can use Python Anaconda, a python environment easily setting packages for data scientists.

## Configuration

svm.py file contains a dictionnary with the predictions to compute. It looks like :

    predict_params = [
      {
        # The model of prediction to execute. The algorithm is done for SVC and LinearSVC
        # Warning : a bug actually make the SVC model not work (see scripts/tools.py comments)
        'model': LinearSVC,
        # Parameter to limit high values to the given value.
        # Here, all data with a value over 4 will be set to 4.
        # Set to 0 to do not process to the cut
        'cut': 4,
        # Number of cross-validation iteration
        # Set to 0 to do not run cross-validation
        'cv': 5,
        # If set to 1, activate the verbose mode in fit and cross-validation methods. Set to 0 do deactivate.
        'verbose': 1
      },
    ]

## Launch

    python svm.py
