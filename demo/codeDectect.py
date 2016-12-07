import os
root=r'G:\python-workplace\my_TextClf_sklearn\my_corpus\训练库\旅游'

def read():
    pass
def filter_short_text():
    list_dir=os.listdir(root)
    for dir in list_dir:
        sub_path=os.path.join(root, dir)
        list_file=os.listdir(sub_path)
        for file in list_file:
            with open(os.path.join(sub_path, file),encoding='utf-8',errors='ignore') as f:
                data=f.read()
            if len(data)<65:
                os.remove(os.path.join(sub_path, file))
def text_transform():
    list_file=os.listdir(root)
    n=0
    for file in list_file:
        if file.split(".")[0].isdigit():
           with open(os.path.join(root, file),encoding='gbk',errors='ignore') as f:
               text=f.read()
           os.remove(os.path.join(root, file))
           with open(os.path.join(root,'text'+file),'w+',encoding='utf-8') as f:
               f.write(text)

text_transform()


