import json
import sys

def mkreport(fun,name,data_path,count = None):
    doc_g = open(data_path,'r',encoding='utf-8')
    exp = fun(doc_g)
    data = []
    print("\nReport of %s begin\n"%name)

    for source,title,pred in exp:
        data.append({
            'source':source,
            'gen':pred,
            'ref':title
        })
        if count is not None:
            sys.stdout.write('\r count:%d'%count)
            if count>0:
                count -= 1
            else:
                break

    jsfile = open(name+'.json','w',encoding='utf-8')
    json.dump(data,jsfile,ensure_ascii=False)
    print("\nReport of %s finished\n"%name)
    return data
