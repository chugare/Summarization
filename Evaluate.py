import nltk
from nltk.translate.bleu_score import sentence_bleu
import json
import rouge
import jieba
import Tool
def generate_BLEU(fname):
    score_sum = 0
    dfile = open(fname,'r',encoding='utf-8')
    data = json.load(dfile)
    count = 0
    for d in data:
        gen = d['gen']
        gen = [i for i in gen]
        ref = d['ref']
        ref = [[i for i in ref]]
        score = sentence_bleu(ref,gen,weights=(1.0,))
        # print(score)
        count += 1
        score_sum += score
    return score_sum/count
def generate_ROUGE(fname):
    score_sum = 0
    dfile = open(fname,'r',encoding='utf-8')
    data = json.load(dfile)
    count = 0
    r = rouge.Rouge()
    sum = None
    for d in data:
        gen = d['gen']
        gen = ' '.join([i for i in gen])
        # gen = [[i for i in gen]]
        ref = d['ref']
        ref = ' '.join([i for i in ref])

        # ref = [[i for i in ref]]



        score = r.get_scores([gen],[ref])[0]
        if sum is None:
            sum = score
        else:
            for t in score:
                for i in score[t]:
                    sum[t][i] += score[t][i]

        print(score)
        count += 1
    avg = {}
    for t in sum:
        avg[t] = {}
        for r in sum[t]:
            avg[t][r] = float(sum[t][r]) / count
        print("%s %f" % (t, avg[t]['r']))
    return score_sum/count
def calcWordNum(gen):
    words = jieba.lcut(gen)
    word_dic = {}
    for w in words:
            word_dic[w] = word_dic.get(w,0)+1
    return len(word_dic)
def calcIdfAverage(gen):
    tfidf = Tool.Tf_idf()
    words = jieba.lcut(gen)
    idfsum = 0
    c = 0
    for w in words:
        if w in tfidf.idf:
            idfsum += tfidf.idf[w]
            c += 1
    return idfsum/c
def generate_ALL(fname):
    score_sum = 0
    dfile = open(fname, 'r', encoding='utf-8')
    data = json.load(dfile)
    count = 0
    r = rouge.Rouge()
    rougeSum = None
    BLEUSum = 0
    wordNumSum = 0
    wordIdfSum = 0.0
    for d in data:
        gen = d['gen']
        wordCount = calcWordNum(gen)
        wordIdf = calcIdfAverage(gen)
        wordNumSum += wordCount
        wordIdfSum += wordIdf
        gen = ' '.join([i for i in gen])
        ref = d['ref']
        ref = ' '.join([i for i in ref])
        score = r.get_scores([gen], [ref])[0]
        if rougeSum is None:
            rougeSum = score
        else:
            for t in score:
                for i in score[t]:
                    rougeSum[t][i] += score[t][i]

        count += 1
        gen = d['gen']
        gen = [i for i in gen]
        ref = d['ref']
        ref = [[i for i in ref]]
        score = sentence_bleu(ref, gen, weights=(1.0,))
        BLEUSum += score

    rougeAvg = {}
    for t in rougeSum:
        rougeAvg[t] = {}
        for r in rougeSum[t]:
            rougeAvg[t][r] = float(rougeSum[t][r]) / count
        print("%s %f" % (t, rougeAvg[t]['r']))
    BLEUAvg = BLEUSum/ count
    wordIdfAvg = wordIdfSum/count
    wordNumAvg = wordNumSum/count
    print('BLEU %f'%BLEUAvg)
    print('WordNUM %f'%wordNumAvg)
    print('WordIDF %f'%wordIdfAvg)
generate_ALL('result.json')
# generate_BLEU('result.json')
# generate_ROUGE('result.json')
