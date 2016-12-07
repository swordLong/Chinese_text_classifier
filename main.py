from TextClf import *
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
def report():
    pass
if __name__ == '__main__':
    # data_set = load_files(r'G:\python-workplace\sougou_corpus\Sogou-C-Train',encoding='utf-8')
    # print(len(data_set.data), len(data_set.filenames), len(data_set.target), len(data_set.target_names))
    # cut_Dataset(data_set,nomial=True)
    data_set=load_jieba_repository(r'仓库\名词性分词.txt')
    # data_set = load_jieba_repository('cut_nomial_test.dat')
    s=time.time()
    train_size_vec = [i * 0.1 for i in range(1, 10)]
    classifiers = ['NB', 'LR', 'RF', 'DT', 'SVM']
    # classifiers = ['NB', 'SVM']
    cm_diags = np.zeros((10, len(train_size_vec), len(classifiers)), dtype=float)  #三维数组 用来放结果：每个分类器在spilt参数下的10个类别准确率
    for n, split in enumerate(train_size_vec):
        for m, classifier in enumerate(classifiers):
            y_pred, y_test = train_predict(data_set, classifier=classifier, use_tf=False, split=split, K=50000)
            # print(metrics.confusion_matrix(y_test, y_pred).diagonal())
            cm_diags[:, n, m] = metrics.confusion_matrix(y_test, y_pred).diagonal()
            # print(cm_diags)
            cm_diags[:, n, m] /= np.bincount(y_test)
            # print(cm_diags)
    #调色板
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    fig, axes = plt.subplots(1, len(classifiers), figsize=(12, 4))
    values = range(10)
    jet = cm = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    # print(scalarMap.get_clim())
    colors=[]
    for idx in range(10):
        colorVal = scalarMap.to_rgba(values[idx])
        colors.append(colorVal)
    #画图
    for m, classifier in enumerate(classifiers):
        for i in range(0,10):
            axes[m].plot(train_size_vec, cm_diags[i, :, m], label=data_set.target_names[i],color=colors[i])
        axes[m].set_title(classifier)
        axes[m].set_ylim(0, 1.1)
        axes[m].set_xlim(0.1, 0.9)
        axes[m].set_ylabel("classification accuracy")
        axes[m].set_xlabel("training size ratio")
        axes[m].legend(loc=4)
    fig.tight_layout()
    plt.show()
    print('totally took %.3f s' % (time.time()- s))
    # save_model(model, '.\model\\'+classifier+'.model')