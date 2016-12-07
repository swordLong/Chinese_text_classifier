import time
import jieba.posseg as pseg
import jieba
from multiprocessing.dummy import Pool as ThreadPool


def contain(list, char):
    for str in list:
        if char.find(str) != -1:
            return True
    return False


def cut_Text(content, nomial=False):
    """
    :param content: string
    :param nomial: if nomial is True,only noun-like words will remain
    :return:a text which format is 'a   b   c   d'
    """
    if nomial:
        text = ''
        words = pseg.cut(content)
        for word in words:
            if contain(['n'], word.flag):
                text = text + ' ' + word.word
        return text.strip()
    else:
        text = ''
        words = jieba.cut(content)
        for word in words:
            text = text + ' ' + word
        return text.strip()


def cut_Dataset(data_set, parrel=False, nomial=False):
    """
    :param data_set:bunch of Dataset
    :param parrel: if it is True,cut dataset in parrel.Windows is not available
    :param nomial: if nomial is True,only noun-like words will remain
    :return:data_set after cutted
    """
    from tqdm import tqdm
    data_cut = []
    start = time.time()
    print('cuting dataset......')
    if parrel:
        p = ThreadPool(9)
        p.map(cut_Text, data_set.data)
        p.close()
        p.join()
    else:
        n=0
        for doc_content in tqdm(data_set.data):
            data_cut.append(cut_Text(doc_content, nomial))
    end = time.time()
    print('cuting  runs %0.2f seconds.' % (end - start))
    data_set.data = data_cut
