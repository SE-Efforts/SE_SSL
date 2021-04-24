from data import Data
from tensorflow.keras.utils import to_categorical
# from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
import numpy as np


def preprocess(df, filename):
    df.drop(['Unnamed: 0', 'bugID'], axis=1, inplace=True)
    _df = df[['s1', 's2', 's3', 's4', 's5', 's6', 's8', 'y']]
    _df['s70'] = df['s7'].apply(lambda x: eval(x)[0])
    _df['s71'] = df['s7'].apply(lambda x: eval(x)[1])
    _df['s72'] = df['s7'].apply(lambda x: eval(x)[2])
    _df['s90'] = df['s9'].apply(lambda x: eval(x)[0])
    _df['s91'] = df['s9'].apply(lambda x: eval(x)[1])

    if filename == 'firefox.csv':
        _df['s92'] = df['s9'].apply(lambda x: eval(x)[2])
        _df['s72'] = df['s7'].apply(lambda x: eval(x)[2])

    x = _df.drop('y', axis=1)
    y = _df['y']

    return x, y


def split_data(filename: str, data: Data, n_classes: int):
    if n_classes == 2:
        if filename == 'firefox.csv':
            data.y_train = data.y_train < 4
            data.y_test = data.y_test < 4
        elif filename == 'chromium.csv':
            data.y_train = data.y_train < 5
            data.y_test = data.y_test < 5
        else:
            data.y_train = data.y_train < 6
            data.y_test = data.y_test < 6
    elif n_classes == 3:
        data.y_train = np.where(data.y_train < 2, 0,
                                np.where(data.y_train < 6, 1, 2))
        data.y_test = np.where(
            data.y_test < 2, 0, np.where(data.y_test < 6, 1, 2))
    elif n_classes == 5:
        data.y_train = np.where(data.y_train < 1, 0, np.where(data.y_train < 3, 1, np.where(
            data.y_train < 6, 2, np.where(data.y_train < 21, 3, 4))))
        data.y_test = np.where(data.y_test < 1, 0, np.where(data.y_test < 3, 1, np.where(
            data.y_test < 6, 2, np.where(data.y_test < 21, 3, 4))))
    elif n_classes == 7:
        data.y_train = np.where(data.y_train < 1, 0, np.where(data.y_train < 2, 1, np.where(data.y_train < 3, 2, np.where(
            data.y_train < 6, 3, np.where(data.y_train < 11, 4, np.where(data.y_train < 21, 5, 6))))))
        data.y_test = np.where(data.y_test < 1, 0, np.where(data.y_test < 2, 1, np.where(data.y_test < 3, 2, np.where(
            data.y_test < 6, 3, np.where(data.y_test < 11, 4, np.where(data.y_test < 21, 5, 6))))))
    else:
        data.y_train = np.where(data.y_train < 1, 0, np.where(data.y_train < 2, 1, np.where(data.y_train < 3, 2, np.where(data.y_train < 4, 3, np.where(
            data.y_train < 6, 4, np.where(data.y_train < 8, 5, np.where(data.y_train < 11, 6, np.where(data.y_train < 21, 7, 8))))))))
        data.y_test = np.where(data.y_test < 1, 0, np.where(data.y_test < 2, 1, np.where(data.y_test < 3, 2, np.where(data.y_test < 4, 3, np.where(
            data.y_test < 6, 4, np.where(data.y_test < 8, 5, np.where(data.y_test < 11, 6, np.where(data.y_test < 21, 7, 8))))))))

    if n_classes > 2:
        data.y_train = to_categorical(data.y_train, num_classes=n_classes)
        data.y_test = to_categorical(data.y_test, num_classes=n_classes)
    return data


if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print(f'Usage: python3 {sys.argv[0]} FILENAME N_CLASSES')
    #     return

    filename = sys.argv[1]
    n_classes = int(sys.argv[2])

    df = pd.read_csv(f'./Bug-Related-Activity-Logs/{filename}')
    x, y = preprocess(df)

    data = Data(*train_test_split(x, y))
    split_data(filename, data, n_classes)

    frac = sum(data.y_train) / len(data.y_train)

    # Some reasonable threshold 0.5 +/- epsilon
    if frac > 0.6:
        # Prepare for WFO
        data.y_train = 1 - data.y_train
        data.y_test = 1 - data.y_test
        frac = 1. - frac

    config = {
        'n_runs': 20,
        'transforms': ['normalize', 'standardize', 'robust', 'maxabs', 'minmax'] * 30,
        'metrics': ['d2h', 'accuracy', 'pd', 'prec', 'pf'],
        'random': True,
        'learners': [],
        'log_path': './log',
        'data': [data],
        'name': filename
    }
    # for _ in range(50):
    #     # If multi-class, use MulticlassDL instead.
    #     if frac < 0.4:  # or some threshold
    #         config['learners'].append(
    #             FeedforwardDL(random={'n_layers': (2, 6), 'n_units': (3, 20)}, n_epochs=50, weighted=True, wfo=True))
    #     else:
    #         config['learners'].append(
    #             FeedforwardDL(random={'n_layers': (2, 6), 'n_units': (3, 20)}, n_epochs=50))
    #
    # dodge = DODGE(config)
    # dodge.optimize()
