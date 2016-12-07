from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import jieba
from os.path import join
from os.path import isdir
from os import listdir
import numpy as np
from sklearn.datasets.base import Bunch


def load_files(container_path, encoding='utf-8', decode_error='strict'):
    target = []
    target_names = []
    filenames = []
    folders = [f for f in sorted(listdir(container_path)) if isdir(join(container_path, f))]
    for label, folder in enumerate(folders):
        target_names.append(folder)
        folder_path = join(container_path, folder)
        documents = [join(folder_path, d)
                     for d in listdir(folder_path)]
        target.extend(len(documents) * [label])
        filenames.extend(documents)

    # convert to array for fancy indexing
    filenames = np.array(filenames)
    target = np.array(target)
    data = []
    for filename in filenames:
        with open(filename, 'r', encoding=encoding, errors=decode_error) as f:
            data.append(f.read())
    # jieba.analyse.set_stop_words('chinese_stopword.txt')
    # jieba.analyse.extract_tags()
    return Bunch(data=data,
                 filenames=filenames,
                 target_names=target_names,
                 target=target)


def cutText(dataset, stop_words=None):  # dataset is a Bunch object,stop word is a list
    data_cut=[]
    if stop_words is not None:
        for text in dataset.data:
            each_cut = []
            each_cut.append(list(jieba.cut(text.strip().replace('\r\n',''))))
            data_cut.append(each_cut)

    return Bunch(data=data_cut,filenames=dataset.filenames,
                        target_names=dataset.target_names,
                        target=dataset.target)


# train_set=datasets.load_files(r'my_corpus\训练库')
train_set = load_files(r'sougou_corpus\sogou-C-Test', encoding='gbk', decode_error='ignore')
print(len(train_set.data), len(train_set.filenames), len(train_set.target), len(train_set.target_names))
data_cut = cutText(train_set,stop_words='stop')
print(len(data_cut.data))
print(data_cut.data[0])
print(data_cut.data[1])
print(data_cut.data[-1])
exit()
f = open('chinese_stopword.txt', 'r', encoding='utf-8-sig')
stop_words = list(f.read().splitlines())
wf_matrix = CountVectorizer(stop_words=stop_words, decode_error='ignore')

train_wf_matrix = wf_matrix.fit_transform(train_set.data)
print(train_wf_matrix.shape)
f.close()

# wf__train_matrix=wf_matrix.transform(train_set.data)
