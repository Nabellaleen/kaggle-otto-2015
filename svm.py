from sys import stderr

from sklearn.svm import SVC, LinearSVC

from scripts.tools import predict

predict_params = [
    # Score kaggle : 13.58967
    # {'model': LinearSVC, 'cut': 4,    'cv': 5, 'verbose': 1},

    # Score kaggle : NA - don't work
    # {'model': SVC,       'cut': 4,    'cv': 5, 'verbose': 1},

    # Score kaggle : 4.37996
    # {'model': LinearSVC, 'cut': 30,   'cv': 5, 'verbose': 1},

    # Score kaggle : NA - don't work
    # {'model': SVC,       'cut': 30,   'cv': 5, 'verbose': 1},

    # Score kaggle : 3.30013
    {'model': LinearSVC, 'cut': None, 'cv': 5, 'verbose': 1},

    # Score kaggle : NA - don't work
    # {'model': SVC,       'cut': None, 'cv': 5, 'verbose': 1},
]

results = []
for predict_param in predict_params:
    try:
        name, logloss = predict(
            model=predict_param['model'],
            cut=predict_param['cut'],
            cv=predict_param['cv'],
            verbose=predict_param['verbose'])
        results.append({
            'name': name,
            'logloss': logloss})
    except Exception as err:
        print(err, file=stderr)

from pprint import pprint
pprint(results)
