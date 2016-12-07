from sklearn import datasets
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
from sklearn import neighbors
from sklearn import svm
from sklearn import ensemble
from sklearn import cluster
# % matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

print(tuple('123'))
exit()


iris = datasets.load_iris()# 载入鸢尾花数据集
train_size_vec = [i*0.1 for i in range(1,9)]
classifiers = [tree.DecisionTreeClassifier,
               svm.SVC,
              ]
cm_diags = np.zeros((3, len(train_size_vec), len(classifiers)), dtype=float) # 用来放结果
for n, train_size in enumerate(train_size_vec):
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(iris.data, iris.target, train_size=train_size)

    for m, Classifier in enumerate(classifiers):
        classifier = Classifier()
        classifier.fit(X_train, y_train)
        y_test_pred = classifier.predict(X_test)
        cm_diags[:, n, m] = metrics.confusion_matrix(y_test, y_test_pred).diagonal()
        cm_diags[:, n, m] /= np.bincount(y_test)
fig, axes = plt.subplots(1, len(classifiers), figsize=(12, 3))

for m, Classifier in enumerate(classifiers):
    axes[m].plot(train_size_vec, cm_diags[2, :, m], label=iris.target_names[2])
    axes[m].plot(train_size_vec, cm_diags[1, :, m], label=iris.target_names[1])
    axes[m].plot(train_size_vec, cm_diags[0, :, m], label=iris.target_names[0])
    axes[m].set_title(type(Classifier()).__name__)
    axes[m].set_ylim(0, 1.1)
    axes[m].set_xlim(0.1, 0.9)
    axes[m].set_ylabel("classification accuracy")
    axes[m].set_xlabel("training size ratio")
    axes[m].legend(loc=4)

fig.tight_layout()
plt.show()
exit()
iris = datasets.load_iris() # 载入鸢尾花数据集
print(iris.target_names)
print(iris.feature_names)
['setosa' 'versicolor' 'virginica']
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

print(iris.data.shape)
print(iris.target.shape)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, train_size=0.7) # 70% 用于训练，30% 用于检验
classifier = linear_model.LogisticRegression()
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)
print(metrics.classification_report(y_test, y_test_pred)) # 真实的 y 和预测的 y