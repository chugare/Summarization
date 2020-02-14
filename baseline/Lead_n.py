import re

def covage(doc_g):
    for line in doc_g:
        title,_,source = line.split('#')
        source_sens_raw = re.split('[。；;]',source)
        source_sens = [''.join(s.split(' ')) for s in source_sens_raw]
        res = ''
        for sen in source_sens:
            segs = sen.split('，')
            if len(res) + len(segs[0])<20:
                res += segs[0]
            else:
                if len(res) == 0:
                    res = segs[0]
                break

        yield ''.join(source.split(' ')),''.join(title.split(' ')),res

def lead(doc_g):

    for line in doc_g:
        title,_,source = line.split('#')
        source_sens_raw = re.split('[，。；;]',source)
        source_sens = [''.join(s.split(' ')) for s in source_sens_raw]


        yield ''.join(source.split(' ')),''.join(title.split(' ')),source_sens[0]

