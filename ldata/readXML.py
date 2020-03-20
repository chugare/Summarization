
import re
import sys
sys.path.append('/home/user/zsm/Summarization')

from ldata.db_util import MysqlWriter



class LCSTSHandler:
    def __init__(self,MR):
        self.data = []
        self.summary = ''
        self.id = ''
        self.content = ''
        self.tag = []
        self.MR = MR
        self.count = 0
    # 元素开始事件处理
    def startElement(self, tag, attributes):
        self.tag.append(tag)
        if tag == "doc":
            self.id = attributes["id"]

    # 元素结束事件处理
    def endElement(self, tag):

        if tag == "doc":
            # self.data.append({
            #     'id':self.id,
            #     'summary': self.summary,
            #     'content': self.content,
            # })
            res = (self.id,self.summary,self.content)
            self.MR.write(res)
            self.count += 1
            if self.count % 1000 == 0:
                print("[I]  %d news inserted"%self.count)
            pass
        self.tag.pop(-1)
        # elif tag == "summary":
        #     pass
        # elif tag == "short_text":
        #     pass

    # 内容事件处理
    def characters(self, content):
        if self.tag[-1] == "summary":
            self.summary = content
        elif self.tag[-1] == "short_text":
            self.content = content


if (__name__ == "__main__"):
    MR = MysqlWriter(buffer_size=100)
    Handler = LCSTSHandler(MR)
    ifile = open("/home/user/zsm/Summarization/lcstsdata/PART_I.txt",'r',encoding='utf-8')
    for line in ifile:
        res = re.findall('<(.*?)>',line)
        if len(res) == 0:
            Handler.characters(line.strip())
        else:
            name = res[0]
            if name.startswith('/'):
                Handler.endElement(name[1:])
            else:
                attrs = res[0].split(" ")
                name = attrs[0]
                attrs = attrs[1:]
                attr = {}
                for i in attrs:
                    atname,atvalue = i.split('=')
                    atvalue = atvalue.strip('"')
                    attr[atname] = atvalue
                Handler.startElement(name,attr)
