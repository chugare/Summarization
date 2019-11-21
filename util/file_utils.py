import os,traceback
def queue_reader(name,path_of_dir):
    if isinstance(name,list):
        try:

            for fname in name:
                file = open(path_of_dir+'/'+fname,'r',encoding='utf-8')
                for line in file:
                    yield line
        except Exception:
            print(traceback.format_exc())
    else:
        files = os.listdir(path_of_dir)
        for fname in files:
            if name in fname:
                try:

                    file = open(path_of_dir+'/'+fname,'r',encoding='utf-8')
                    for line in file:
                        yield line
                except Exception:
                    print(traceback.format_exc())


if __name__ == '__main__':
    read = queue_reader("E","/home/user/zsm/Summarization/data")
    for i in read:
        pass