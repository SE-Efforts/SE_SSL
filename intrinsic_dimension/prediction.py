# from __future__ import division
import numpy as np
import glob, os, sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
import time
import warnings
warnings.filterwarnings("ignore")
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.utils import to_categorical
# from keras.callbacks import EarlyStopping
import pdb
from estimate_intrinsic_dim import Solver as IntrinDimSolver
from utils import ConfigBase

""" NOTES:
            - requires Python 3.0 or greater
"""

__author__ = 'Sherry'


class Config(ConfigBase):
    def __init__(self):
        super(Config, self).__init__()
        self.est_config = "="*6 + " Intrinsic Dim Estimation of real dataset " + "="*6
        self.log_dir = "exp/realcase/log"
        self.log_filename = "est.log"
        self.logrs = "-4:0:20"  # from -4 to 0 with 20 steps
        

class DatasetWrapper():
    def __init__(self, dataset, config):
        self.samples = dataset

    def __len__(self):
        return self.samples.shape[0]


def open(path):
        testFiles = path + "/test_set/totalFeatures5.csv"
        trainingFiles = glob.glob(path + "/training_set/*.csv")
        list_training = []  # training set list
        list_header = []    # get the list of headers of dfs for training & testing
        for file in trainingFiles:
                df = pd.read_csv(file, index_col=None, header=None, skiprows=0)
                head = df.iloc[0]
                head.tolist()
                list_header.append(head)    # list of header for training set
                list_training.append(df)    # list of training set

        testing = pd.read_csv(testFiles, index_col=None, header=None, skiprows=0)
        list_header.append(testing.iloc[0].tolist())
        return testing, list_training, list_header


def df_get(df):
        """
        :param df: a data frame with header
        :return: get rid of the header of data frame
        """
        header = df.iloc[0]
        # Create a new variable called 'header' from the first row of the dataset

        # Replace the dataframe with a new one which does not contain the first row
        df = df[1:]

        # Rename the dataframe's column values with the header variable
        df = df.rename(columns=header)
        h1 = list(df.columns.values)    # get the value of header of the df
        return df, h1


def common_get(list_header):
        """
        :param list_header: list of training & testing headers
        :return: common header
        """

        golden_fea = ["F116", "F115", "F117", "F120", "F123", "F110", "F105", "F68", "F101", "F104", "F65", "F22",
                                    " F94", "F71", "F72", "F25", "F3-", "F15", "F126", "F41", "F77"]
        # golden_fea = ["F116", "F115", "F117", "F120", "F123", "F110", "F105", "F68", "F101", "F104", "F65", "F22", "F20",
        #                               "F21", " F94", "F71", "F72", "F25", "F3-", "F15", "F126", "F41", "F77"]
        # mentioned in Table 4 from Dr. Wang's paper: Commonly-selected features (RQ1)

        golden_fea.append("category")
        golden_list = []
        count_list = []
        for header in list_header:
                golden = []
                count = 0
                for i in header:
                        if i.startswith(tuple(golden_fea)):
                                count += 1
                                golden.append(i)
                # print("number of golden fea:", count)
                count_list.append(count)
                golden_list.append(golden)

        common = set(golden_list[0])
        for s in golden_list[1:]:
                common.intersection_update(s)
        return common


def trim(df, common):
        df1, header = df_get(df)
        for element in header:
                if element not in common:
                        df1 = df1.drop(element, axis=1)
        df_trim = df1
        return df_trim


def merge1(testing, list1, list2):
        # not in use
        """
        :param testing: testing set
        :param list1: list of training data frame
        :param list2: list of headers of training set
        :return: get rid of uncommon parts of 5 version in training set
        """

        head5 = testing.iloc[0].tolist()
        holder = head5
        for i in list2:
                common = intersect(i, holder)
                holder = common

        common_header = common

        df, header = df_get(list1[0])  # ONLY USE THE VERSION 4 FOR TRAINING
        for element in header:
                if element not in common_header:
                        df = df.drop(element, axis=1)
        df_merge = df

        # df_merge = pd.concat([i for i in list1], ignore_index=True, sort=True)

        return df_merge, common_header


def is_number(df):
        '''
        :param df: input should be training_x, testset_x(type: data frame)
        :return: return is index of numeric features
        '''

        index = []
        position = 0
        for i in range(len(df.iloc[0])):
                s = df.iloc[0, i]
                try:
                        float(s)    # for int, long and float
                        index.append(i)
                except ValueError:
                        position += 1
        return index


def preprocess1(Y, X):
        index = []
        label = []
        # index_target = []
        for i in range(0, len(Y)):
                # y = Y[0][i]
                y = Y.iloc[i]
                if y == "close":
                        # y = "yes"
                        y = 1
                        # index_target.append(i)
                elif y == "open":
                        # y = "no"
                        y = 0
                elif y == "deleted":
                        index.append(i)  # index is a list of index for deleted samples
                label.append(y)

        for i in sorted(index, reverse=True):  # delete samples with deleted label
                del label[i]
                del X[i]

        # X_text = []                   # transfer samples of X from list to str pd.Series
        # for i in X:
        #           X_text.append(pd.Series(i).str.cat(sep=','))
        #           X = X_text  # list(type)
        return label, X


def ratio_down(Y, X, perc):
        # decrease the ratio of target in training set
        index_target = []
        for i in range(0, len(Y)):
                y = Y[i]
                if y == 1:
                        index_target.append(i)

        percent = perc
        print('In training set:')
        print('number of target is', len(index_target) - round(len(index_target) * percent))
        print('number of non-target is', len(Y) - len(index_target))
        print('ratio of target is', (len(index_target) - round(len(index_target) * percent)) /
                    (len(Y) - round(len(index_target) * percent)))

        for i in sorted(index_target[:round(len(index_target) * percent)], reverse=True):
                # remove the percent% of target samples
                del Y[i]
                # del X[i]              # error! cannot do this to dataframe
                X = X.drop(X.index[i])
        ratio = (len(index_target) - round(len(index_target) * percent)) / (len(Y) - round(len(index_target) * percent))

        return Y, X, ratio


def training_down(Y, X, perc):
        # decrease the ratio of target in training set
        index_target = []
        for i in range(0, len(Y)):
                y = Y[i]
                if y == 1:
                        index_target.append(i)

        percent = perc
        print('In training set:')
        print('number of target is', len(index_target) - round(len(index_target) * percent))
        print('number of non-target is', len(Y) - len(index_target))
        print('ratio of target is', (len(index_target) - round(len(index_target) * percent)) /
                    (len(Y) - round(len(index_target) * percent)))

        for i in sorted(index_target[:round(len(index_target) * percent)], reverse=True):
                # remove the percent% of target samples
                del Y[i]
                # del X[i]              # error! cannot do this to dataframe
                X = X.drop(X.index[i])
        ratio = (len(index_target) - round(len(index_target) * percent)) / (len(Y) - round(len(index_target) * percent))

        return Y, X, ratio


def one_hot(df, index_num):
        """
        :param df: training_x or testset_x, type: data frame
        :param index_num: the index list of numerical features
        :return:
        """
        lb = LabelBinarizer()
        list_len = list(range(len(df.iloc[0])))
        index_onehot = list(set(list_len) - set(index_num))
        for i in index_onehot:
                df.iloc[:, i] = lb.fit_transform(df.iloc[:, 26]).tolist()
        return df


def distribution(y):
        '''
        :return:    get the distribution of y
        '''
        target_count = 0
        for i in y:
                if i == 1:
                        target_count += 1
        ratio = target_count / len(y)

        return ratio


def data_preparing(path, stop_at, perc, seed=0):
        testing, list_training, list_header = open(path)
        # merge, common = merge1(testing, list_training, list_header)
        common_header = common_get(list_header)
        np.random.seed(seed)

        # training set
        training_trim = trim(list_training[0], common_header)  # ONLY USE THE VERSION 4 FOR TRAINING

        # testing set
        testing_trim = trim(testing, common_header)
        # testing_trim.iloc[0]   to check the type of each feature

        # training set
        training_x = training_trim.iloc[:, :-1]  # pandas.core.frame.DataFrame
        training_y = training_trim.iloc[:, -1]  # pandas.core.series.Series

        # testing set
        testset_x = testing_trim.iloc[:, :-1]
        testset_y = testing_trim.iloc[:, -1]

        # remove the samples in training and test set with label "deleted"
        training_y, training_x = preprocess1(training_y, training_x)
        testset_y, testset_x = preprocess1(testset_y, testset_x)
        training_y, training_x, ratio = ratio_down(training_y, training_x, perc=perc)

        # using one-hot to preprepossing the label
        # training_y = to_categorical(training_y)
        # testset_y1 = to_categorical(testset_y)

        le = preprocessing.LabelEncoder()
        # kaggle's forums where to find valuable information

        # normalize the x for training and test sets
        min_max_scaler = preprocessing.MinMaxScaler()
        scaler = MinMaxScaler()

        testset_x = np.asarray(testset_x)  # convert dataframe into numpy.array to normalize
        # ########
        testset_x = min_max_scaler.fit_transform(testset_x)

        training_x = np.asarray(training_x)  # convert dataframe into numpy.array to normalize

        # ########   linear svm needs normalization to reduce the running cost on server
        training_x = min_max_scaler.fit_transform(training_x)

        return training_x


def model_init(testset_x):
        model = Sequential()    # creat model

        # get number of colums or features i testset data
        n_cols = testset_x.shape[1]

        # add the layers for model
        model.add(Dense(30, activation='relu', input_shape=(n_cols,)))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        # compile model using accuracy to measure the performance
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        # model.compile(optimizer='adam', loss='categorical_crossentropy',
        #                               metrics=['accuracy'])

        return model


def main(config):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        projects = ['derby', 'mvn', 'lucence', 'phoenix', 'cass', 'jmeter', 'tomcat', 'ant', 'commons']
        # projects = ['cass']
        stopats = [1]
        # perc_lists = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        perc_lists = [0]
        # how many percent of targets to cut down in training data to cut down
        for stopat_id in stopats:
                # print("----------threshold stop at----------:", stopat_id)
                for project in projects:
                        start_time = time.time()
                        # path = r'data/total_features/' + project
                        path = r'data/' + project
                        print("-----------------" + project + "----------------------")
                        AUC = []
                        cost = []
                        Acc = []
                        for perc in perc_lists:
                                training_data = data_preparing(path, stop_at=stopat_id, perc=perc,
                                                                         seed=int(time.time() * 1000) % (2 ** 32 - 1))
                                # intrinDimEstimation
                                print(training_data.shape)
                                dataset = DatasetWrapper(training_data, config)
                                print(dataset.samples.shape)
                                print(type(dataset.samples))
                                print(">> creating intrinsic dimension solver")
                                solver = IntrinDimSolver(dataset, config)
                                print(">> solving...")
                                solver.show_curve(config.logrs, version=1)
                                # version = 1 or 2 : L1 or L2
                                print(">> task finished")

if __name__ == "__main__":
    from utils import Logger
    config = Config()
    config.parse_args()
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    sys.stdout = Logger('{0}/{1}'.format(config.log_dir, config.log_filename))
    config.print_args()
    main(config)
