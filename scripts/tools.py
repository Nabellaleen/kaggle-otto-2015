from datetime import datetime
from sys import stdout, stderr

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

def log_loss_scorer(estimator, X, y):
    prob = estimator.predict_proba(X)
    return log_loss(y, prob)

def predict(model, cut=None, cv=None,
            verbose=0):
    model_name = model.__name__
    cut_name = 'cut{val}'.format(val=cut) if cut else 'nocut'
    file_name = '{model_name}_{cut_name}_{now_time}'.format(
        model_name=model_name,
        cut_name=cut_name,
        now_time=datetime.now().strftime('%Y%m%d-%H%M%S'))

    print('### Start to predict {file_name}'.format(file_name=file_name),
        file=stdout)

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_submission = pd.read_csv("data/sampleSubmission.csv")
    training_labels = LabelEncoder().fit_transform(train['target'])
    train_features = train.drop('target', axis=1)

    if cut:
        # SVMs tend to like features that look similar to ~ N(0,1), so let's stabilise
        # the long tails
        train_features[train_features > cut] = cut

    # Build model
    model_instance = model(verbose=verbose)
    estimator = model_instance.fit(
        train_features, training_labels)

    # TODO : scores has not the same shape for SVC and LinearSVC model
    # So the predictions_normalized has to be computed with another method
    # in the SVC case to have the asked shape
    scores = estimator.decision_function(X=test)
    predictions = 1.0 / (1.0 + np.exp(-scores))
    row_sums = predictions.sum(axis=1)
    predictions_normalised = predictions / row_sums[:, np.newaxis]

    # create submission file
    prediction_DF = pd.DataFrame(
        predictions_normalised,
        index=sample_submission.id.values,
        columns=sample_submission.columns[1:])
    prediction_DF.to_csv(
        'results/{file_name}.csv'.format(file_name=file_name),
        index_label='id')

    try:
        logloss = log_loss_scorer(estimator, train_features, training_labels)
        print('logloss: {logloss}'.format(logloss=logloss),
            file=stdout)
    except AttributeError as err:
        print('model score - AttributeError: {0}'.format(err),
            file=stderr)
        logloss = -1

    print('### Stop to predict {file_name}'.format(file_name=file_name),
        file=stdout)

    if cv:
        print('### Start to cross_validation {file_name}'.format(file_name=file_name),
            file=stdout)

        # Cross validation
        from sklearn import cross_validation

        try:
            results = cross_validation.cross_val_score(
                estimator,
                train_features,
                training_labels,
                scoring=log_loss_scorer,
                cv=cv,
                verbose=verbose)
            print('Accuracy: %0.2f (+/- %0.2f)' % (results.mean(), results.std() * 2),
                file=stdout)
        except AttributeError as err:
            print('cross validation - AttributeError: {0}'.format(err),
                file=stderr)

        print('### Stop to cross_validation {file_name}'.format(file_name=file_name),
            file=stdout)

    return file_name, logloss
