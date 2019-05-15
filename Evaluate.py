import nltk
from nltk.translate.bleu_score import sentence_bleu
import json
def generate_BLEU(fname):
    dfile = open(fname,'r',encoding='utf-8')
    data = json.load(dfile)
    for d in data:
        gen = d['gen'].split(' ')
        gen = ''.join
        ref = [''.join(d['ref'][0].split(' '))]
        score = sentence_bleu(ref,gen)
        print(score)
generate_BLEU('result.json')