# 借助主题模型以及词性关系进行文本生成的实验

想象一下我们的大脑在说一段话的过程中，需要考虑很多因素，包括语法以及单词的含义是否合适。而不仅仅是针对单词本身的考量，所以我希望在新的模型中，考虑单词的类型以及单词主题信息。通过单词上下文，词性信息以及主题信息三个方面的联合概率，实现每一步的单词生成。


## 主题模型

使用 `gensim`库中的`Ldamodel`来进行lda分析，在调用的时候需要先将原来文本输入到`Dictionary`中，将单词变成id才能进行lda的训练。

lda训练的过程增加监控的方式是加入使用`python`的`logging`模块，将`logging` 的设置改为`INFO`模式就可以看到训练的过程

```python
text = open(source,'r',encoding='utf-8')
text_set = [line.split(' ') for line in text]
dic = corpora.Dictionary(text_set)
corpus = [dic.doc2bow(line) for line in text_set]
dic.save('lda_dic')
lda = models.LdaModel(corpus,num_topics=30,passes=100)

```


## 词性关系

## 上下文生成模型
