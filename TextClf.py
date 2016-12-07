from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn.externals import joblib
import time
from sklearn import cross_validation
import numpy as np
from sklearn import metrics

import os

def load_model(model_name):
    '''
    load a model
    '''
    s=time.time()
    print('load a model where %s' % (os.path.join(os.getcwd(), model_name)))
    print('loading model.......')
    model= joblib.load(model_name)
    print('loading took %.2f s' % (time.time() - s))
    return model


def save_model(model_name, clf):
    s = time.time()
    print('save as model where %s' % (os.path.join(os.getcwd(), model_name)))
    print('saving model.......')
    joblib.dump(model_name, clf)
    print('saving took %.2f s' % (time.time() - s))


def load_files(path, encoding='gbk'):
    """
    :param filename:
     structure such as the following:
        container_folder/
            category_1_folder/
                file_1.txt
                file_2.txt
                ...
                file_42.txt
            category_2_folder/
                file_43.txt
                file_44.txt
    :param encoding:
    :return: Bunch object
    """
    return datasets.load_files(path, encoding=encoding, decode_error='ignore', shuffle=False)


def save_jieba_repository(filename, bunch):
    """
    存储分词后的对象
    :param filename: relative path
    :param bunch: Bunch obj
    :return: none
    """
    print('Save as repository where %s' % (os.path.join(os.getcwd(), filename)))
    with open(filename, 'wb') as f:
        pickle.dump(bunch, f)

def load_jieba_repository(filename):
    """
    加载分词文本
    :param filename: relative path
    :return: Bunch obj
    """
    s = time.time()
    print('load a repository where %s' % (os.path.join(os.getcwd(), filename)))
    print('loading dataset......')
    with open(filename, 'rb') as f:
        bunch= pickle.load(f)
        print('loading took %.2f s' % (time.time() - s))
        return bunch

test_classifiers = ['NB', 'LR', 'RF', 'DT', 'SVM', 'SVMCV', 'GBDT','KNN']

def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


def ex_feature(train_set,test_set,t_train,t_test,hash=False,use_tf=False,K=2000):
    '''
    extract feature from train_set,test_set,using term frequency or tf-idf.
    And a stop_word.txt is necessary
    :param train_set:numpy array or sparse matrix of shape [n_samples,n_features]
                    Training data
    :param test_set:numpy array or sparse matrix of shape [n_samples,n_features]
                    Training data
    :param t_train:numpy array of shape [n_samples, n_targets]
                    Target values
    :param t_test:numpy array of shape [n_samples, n_targets]
                    Target values
    :param hash: use HashingVectorizer
    :param use_tf: use term frequency to descend dimensions
    :param K:select k best features based on cki2,only used if ``use_tf == 'False'``
    :return:train_Set and test_set after extracting features
    '''
    with open('chinese_stopword.txt', 'r', encoding='utf-8-sig') as f:
        stop_words = list(f.read().splitlines())
        data_train_size_mb = size_mb(train_set)
        data_test_size_mb = size_mb(test_set)
        start_time = time.time()

        print('extracting features......')
        if hash:
            from sklearn.feature_extraction.text import HashingVectorizer
            vectorizer = HashingVectorizer(non_negative=True)
            x_train = vectorizer.fit_transform(train_set)
            x_test = vectorizer.fit_transform(test_set)
        else:
            tfidf_transformer = TfidfTransformer()
            if use_tf:
                vectorizer = CountVectorizer(max_features=K,stop_words=stop_words, decode_error='strict')
                x_train_tf_matrix = vectorizer.fit_transform(train_set)
                x_train = tfidf_transformer.fit_transform(x_train_tf_matrix)
                x_test_tf_matrix = vectorizer.transform(test_set)#共用一个vectorizer
                x_test = tfidf_transformer.fit_transform(x_test_tf_matrix)
            else:
                from sklearn.feature_selection import SelectKBest
                from sklearn.feature_selection import chi2
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(stop_words=stop_words)
                x_train_tfidf_matrix = vectorizer.fit_transform(train_set)
                x_test_tfidf_matrix = vectorizer.transform(test_set)
                chi2=SelectKBest(chi2, k=K)
                x_train = chi2.fit_transform(x_train_tfidf_matrix, t_train)
                x_test = chi2.transform(x_test_tfidf_matrix)

        end_time=time.time()

        print('extract features took %.2f s  at %0.2fMB/S' % ( (time.time() - start_time), (data_train_size_mb+data_test_size_mb) / (end_time-start_time)))
        return x_train, x_test

def split_dataset(data_set,split=0.5):
    '''
    According to 'spilt',split the dataset to train_set and test_set
    :param data_set: a Bunch object
    :param split: integer
    :return: x_train, x_test, y_train, y_test:Training data and target values
    '''
    print('spilting dataset......')
    start_time = time.time()
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(data_set.data, data_set.target,
                                                                       test_size=split, random_state=0)
    print('spilting took %.2f s' % (time.time() - start_time))
    # train_set=(x_train,y_train)
    # test_set=(x_test,y_test)
    # return train_set,test_set
    return x_train, x_test, y_train, y_test

def train_predict(data_set, classifier=None,report=False,K=2000,use_tf=False,split=0.5):
    '''

    :param data_set: 数据集
    :param classifier: 指定一个分类器
    :param report: 打印报告
    :param K: 使用tf-idf降维至前K维,仅当use_tf为假时有效
    :param use_tf: 使用词频降维
    :param split: 切分参数
    :return: 返回一个训练好的分类器
    '''
    x_train, x_test, y_train, y_test=split_dataset(data_set,split=split)
    x_train,x_test = ex_feature(x_train,x_test,y_train,y_test,K=K,use_tf=use_tf)
    print("train_set: n_samples: %d, n_features: %d" % x_train.shape)
    print("test_set:  n_samples: %d, n_features: %d" % x_test.shape)
    print('************************ %s ***********************' % (classifier))

    print('training...')
    start_time = time.time()
    model = classifiers[classifier](x_train, y_train)
    print('training took %.2f s' % (time.time() - start_time))

    print('testing...')
    start_time = time.time()
    predicted = model.predict(x_test)
    print('testing took %.2f s' % (time.time() - start_time))

    print('accuracy: %.3f' % np.mean(predicted == y_test))
    print('confusion matrix is：')
    # for i in data_set.target:
    #     print()
    cm=metrics.confusion_matrix(y_test, predicted)
    import csv
    print(metrics.confusion_matrix(y_test, predicted))
    with open ('仓库\\'+'confusion_matrix_'+str(split)+classifier+'.csv','w') as f:
        writer = csv.writer(f, dialect="excel")
        temp=[' ']
        temp.extend([name for name in data_set.target_names])
        writer.writerow(temp)
        for i, line in enumerate(cm):
            temp=[]
            temp.append(data_set.target_names[i])
            temp.extend([i for i in line])
            writer.writerow(temp)
    print(metrics.classification_report(y_test, predicted))
    print('************************Done*******************************')
    return predicted,y_test


def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model

# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import LinearSVC
    model =LinearSVC(dual=False,multi_class='crammer_singer',tol=0.01,C=10)
    # model=SVC(kernel='linear',cache_size=2000,decision_function_shape='ovr',gamma=0.001,C=1)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC
    model = LinearSVC(multi_class='ovr',C=1,tol=0.01)
    param_grid = {'dual':[False,True]}
    # param_grid={'dual':[False,True],'C':[1,10,100],'tol':[1e-2,1e-3,1e-1]}
    g_clf = GridSearchCV(model, param_grid, n_jobs=-1, verbose=1)
    g_clf=g_clf.fit(train_x, train_y)
    best_parameters, score, _ = max(g_clf.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(param_grid.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    print(score)
    return g_clf

classifiers = {'NB': naive_bayes_classifier,
               'LR': logistic_regression_classifier,
               'RF': random_forest_classifier,
               'DT': decision_tree_classifier,
               'SVM': svm_classifier,
               'SVMCV': svm_cross_validation,
               'GBDT': gradient_boosting_classifier,
               'KNN':knn_classifier,
               }
