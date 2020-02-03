
import sys,threading,queue
sys.path.append('/home/user/zsm/Summarization')
import tensorflow as tf
import numpy as np
from data_util.tokenization import tokenization
from util.file_utils import queue_reader
from model.RNNs2s import RNNS2Smodel,create_padding_mask

import time
from interface.NewsInterface import NewsBeamsearcher,NewsPredictor


MODEL_PATH = './RNNs2s'
DICT_PATH = '/root/zsm/Summarization/news_data_r/NEWS_DICT_R.txt'
DATA_PATH = '/home/user/zsm/Summarization/news_data'

D_MODEL = 200
ENCODE_SIZE = 256
ENCODER_NUM = 3
DECODER_SIZE = 256
DECODER_NUM = 3
VOCAB_SIZE = 100000


def build_input_fn(name,data_set,batch_size = 1,input_seq_len = 1000,output_seq_len = 100):

    def generator():
        tokenizer = tokenization(DICT_PATH,DictSize=100000)
        source_file = queue_reader(name,data_set)
        for line in source_file:
            try:
                example = line.split('#')
                title = example[0]
                desc = example[1]
                content = example[2]
                title = title.split(' ')
                content = content.split(' ')
                source_sequence = tokenizer.tokenize(content)
                source_sequence = tokenizer.padding(source_sequence,input_seq_len)
                title_sequence = tokenizer.tokenize(title)
                title_sequence = tokenizer.padding(title_sequence,output_seq_len)
                title_sequence.insert(0,2)

                # title_context.append([0]*context_le)

                feature = {
                    'source':source_sequence,
                    'last_word': 0,
                    'title':title_sequence,
                }
                #     yield feature,label
                yield feature,0
            except Exception:
                continue
    def input_fn():
        ds = tf.data.Dataset.from_generator(generator=generator,output_types=({'source':tf.int64,'last_word':tf.int64,'title':tf.int64},tf.int64),output_shapes=({'source':[input_seq_len],'last_word':[],'title':[output_seq_len+1]},[]))
        ds = ds.shuffle(8192).batch(batch_size).cache().repeat()
        return ds
    return input_fn





def build_model_fn(lr,d_model, input_vocab_size,encoder_size,encoder_layer_num,decoder_size,decoder_layer_num):

    learning_rate = lr
    #
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    #     from_logits=True, reduction='none')
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        # loss_ = loss_object(real, pred)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)
    def accuracy_function(real,pred):
        train_accuracy = tf.keras.metrics.sparse_categorical_accuracy(real, pred)
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        mask = tf.cast(mask, dtype=train_accuracy.dtype)

        count = tf.reduce_sum(mask)+1
        train_accuracy = tf.reduce_sum(train_accuracy * mask) / count
        return train_accuracy


    def model_fn(features,labels,mode,params=None):

        # source = features['source']
        # context = features['context']

        global_step = tf.compat.v1.train.get_or_create_global_step()
        source = features['source']
        last_word = features['last_word']
        class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, d_model, warmup_steps=4000):
                super(CustomSchedule, self).__init__()

                self.d_model = d_model
                self.d_model = tf.cast(self.d_model, tf.float32)

                self.warmup_steps = warmup_steps

            def __call__(self, step):
                step = tf.cast(step,tf.float32)
                arg1 = tf.math.rsqrt(step)
                arg2 = step * (self.warmup_steps ** -1.5)

                return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        training = mode == tf.estimator.ModeKeys.TRAIN

        RNNS2S_model = RNNS2Smodel(d_model=d_model,input_vocab_size=input_vocab_size,
                                   encoder_size = encoder_size,encoder_layer_num=encoder_layer_num,
                                   decoder_size=decoder_size,decoder_layer_num=decoder_layer_num)


        if mode == tf.estimator.ModeKeys.TRAIN:
            title = features['title']
            title_real = title[:,1:]
            title_input = title[:,:-1]
            mask = create_padding_mask(title_real)

            prediction, decoder_input = RNNS2S_model(source,None,title_real,mode,None,mask = mask)
            prediction = prediction - tf.expand_dims(tf.reduce_max(prediction,-1),-1)

            loss = loss_function(title_real,prediction)
            learning_rate = CustomSchedule(d_model)(global_step)
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate,beta1=0.9, beta2=0.98,
                                                         epsilon=1e-9)
            train_accuracy = accuracy_function(title_real, prediction)
            grads = optimizer.compute_gradients(loss)
            for grad, var in grads:
                tf.summary.histogram(var.name + '/gradient', grad,global_step)
            grad, var = zip(*grads)
            tf.clip_by_global_norm(grad, 0.5)
            grads = zip(grad, var)
            decoder_states_m = tf.constant(0)
            decoder_states_c = tf.constant(0)
            train_op = optimizer.apply_gradients(grads, global_step=global_step)

            tf.summary.scalar('accuracy',train_accuracy)
        else:

            mask = tf.zeros(shape=tf.shape(last_word))
            state_m = features['state_m']
            state_c = features['state_c']

            prediction,decoder_states = RNNS2S_model(source,last_word,None,mode,[state_m,state_c],mask = mask)
            decoder_states_m,decoder_states_c = decoder_states
            prediction = prediction - tf.expand_dims(tf.reduce_max(prediction,-1),-1)

            loss = tf.constant(0)
            train_accuracy = tf.constant(0)
            train_op = tf.no_op()

        class TransformerRunHook(tf.estimator.SessionRunHook):
            def __init__(self):
                self.count = 0
                self.start_time  = time.time()
                self.ctime = time.time()
            def before_run(self, run_context):
                return tf.estimator.SessionRunArgs({'loss':loss,'accuracy':train_accuracy,'global_step':global_step,
                                                    'learning_rate':learning_rate,'PRED':prediction,'title':title_real,
                                                    'decoder_state':decoder_states_c})

            def after_run(self, run_context, run_values):

                self.count += 1
                # a = np.mean(run_values.results['accuracy'])
                a = run_values.results['accuracy']
                def statis(pred):
                    c_map = {}
                    pred = np.reshape(pred,[-1])
                    for w in pred:
                        c_map.setdefault(w,0)
                        c_map[w] += 1
                    return c_map

                if run_values.results['global_step'] % 5 == 0:
                    ntime = time.time()
                    dtime = ntime - self.ctime
                    self.ctime = ntime
                    print("Batch {0} : loss - {1:.3f} :lr - {5:.2e} : accuracy - {2:.3f} : time_cost - {3:.2f} : all_time_cost - {4:.2f}".format(
                        run_values.results['global_step'],run_values.results['loss'],a,dtime,ntime-self.start_time,run_values.results['learning_rate']))
                    pred = np.argmax(run_values.results['PRED'],-1)
                    pred_o = statis(pred)
                    # print(run_values.results['PRED'])

                pass


        return tf.estimator.EstimatorSpec(mode,{'target':prediction,'state_c':decoder_states_c,'state_m':decoder_states_m},loss,train_op,training_hooks=[TransformerRunHook()])

    return model_fn


class S2SBeamSearcher(NewsBeamsearcher):
    def __init__(self,dataset,tokenizer,topk,predictor,context_len,max_count):
        super(S2SBeamSearcher,self).__init__(dataset,tokenizer,topk,predictor,max_count)
        self.context_len = context_len



    def get_pred_map(self,pred):
        return pred['target'][:,0,:],pred['state_c'],pred['state_m']


    def get_input_fn(self):
        def data_generator():
            searcher = self
            while len(searcher.gen_result) < searcher.max_count:
                state_c,state_m = searcher.state.get()
                # buffer_lock.wait(2)
                yield {'source':searcher.source_input,'state_c':state_c,'state_m':state_m,'last_word':10}, tf.constant(0)

        def input_fn():
            return tf.data.Dataset.from_generator(generator=data_generator,output_types=({'source':tf.int64,'last_word':tf.int64,'state_c':tf.float32,'state_m':tf.float32},tf.int64),
                                                  output_shapes=({'source':[1000],'last_word':[],'state_c':[DECODER_NUM,DECODER_SIZE],'state_m':[DECODER_NUM,DECODER_SIZE]},[])).batch(self.topk)
        return input_fn
    def do_search_mt(self,max_step,estimator,rp_core = None):
        # buffer_lock = threading.RLock()
        # # result_lock = threading.RLock()
        # buffer_lock = threading.Condition(1)
        # result_lock = threading.BoundedSemaphore(self.topk)

        self.state = queue.Queue(1)

        def fill_data(searcher):
            while len(searcher.gen_result) < searcher.max_count:

                if len(searcher.buffer) == 0 or searcher.gen_len == max_step:
                    if searcher.gen_len == max_step:
                        searcher.gen_result.append((searcher.buffer[-1][0],searcher.title_input,searcher.source_input))

                        print('第{0}步生成的内容：'.format(len(searcher.gen_result)))
                        print(searcher.tokenizer.get_sentence(searcher.buffer[-1][0]))
                    source, title = next(searcher.dataset)
                    searcher.source_input = source[:]
                    searcher.title_input = title[:]


                    searcher.buffer = []
                    for i in range(searcher.topk):
                        searcher.buffer.append(([],0))
                    searcher.gen_len = 0
                    searcher.next_topk.empty()
                    searcher.state.empty()
                    state_c = np.zeros([DECODER_NUM,DECODER_SIZE],np.float32)
                    state_m = np.zeros([DECODER_NUM,DECODER_SIZE],np.float32)
                else:

                    tmp_buffer = []
                    buffer = searcher.buffer
                    next_topk , state_c , state_m = searcher.next_topk.get()
                    if searcher.gen_len == 0:
                        candidate,score = buffer[0]

                        for i, n in enumerate(next_topk[0]):
                            ts = next_topk[0][n]
                            tc = candidate[:]
                            if n==1:
                                tc.append(searcher.source_input[0])
                            else:
                                tc.append(n)
                            tmp_buffer.append((tc,score+ts))

                    else:
                        for i, val in enumerate(buffer):
                            candidate,score = val
                            for n in next_topk[i]:
                                ts = next_topk[i][n]
                                tc = candidate[:]

                                if tc[-1] == 1:
                                    tc.append(1)
                                    ts = score
                                else:
                                    tc.append(n)
                                    ts = score + ts

                                # tc.append(searcher.title_input[searcher.gen_len])

                                tmp_buffer.append((tc,ts))

                    tmp_buffer = sorted(tmp_buffer,key=lambda x:x[1])
                    tmp_buffer = tmp_buffer[-searcher.topk:]
                    searcher.buffer = tmp_buffer
                    searcher.gen_len += 1
                #
                # if searcher.gen_len>0:
                #     print('第{0}步生成的内容：'.format(searcher.gen_len))

                if len(searcher.buffer) > 0:
                    c = np.sum([1 if len(l[0])>0 and l[0][-1] == 1 else 0 for l in searcher.buffer])
                else:
                    c = 0
                if c == searcher.topk:
                    # 用于判定是否是第一次的生成，因为后续的生成都是 c * c 的数量
                    searcher.gen_len = max_step
                else:
                    searcher.state.put([state_c,state_m])

        pred = estimator.predict(self.get_input_fn(),['target','state_c','state_m'],yield_single_examples=False)

        def get_next(searcher):

            while len(searcher.gen_result) < searcher.max_count:
                try:
                    res_v = []
                    res = next(pred)
                    res,state_c,state_m = searcher.get_pred_map(res)
                    for i,v in enumerate(res):
                        vmap = v
                        if rp_core is not  None:
                            rp_core.do()
                        sort_res = np.argsort(vmap)[-searcher.topk:]
                        map_res = {}
                        for k in sort_res:
                            map_res[k] = vmap[k]
                        res_v.append(map_res)


                    searcher.next_topk.put((res_v,state_c,state_m))
                except StopIteration:
                    return

        producer = threading.Thread(target=fill_data,args=(self,))
        consumer = threading.Thread(target=get_next,args=(self,))
        producer.start()
        consumer.start()

        producer.join()
        consumer.join()



if __name__ == '__main__':

    def train():
        model_fn = build_model_fn(lr = 0.01,d_model=D_MODEL,input_vocab_size=VOCAB_SIZE,encoder_size=ENCODE_SIZE,encoder_layer_num=ENCODER_NUM,
                                  decoder_size=DECODER_SIZE,decoder_layer_num = DECODER_NUM)
        estimator = tf.estimator.Estimator(model_fn,model_dir=MODEL_PATH,)
        input_fn = build_input_fn("NEWS", DATA_PATH,batch_size=32,input_seq_len=1000,output_seq_len=100)

        estimator.train(input_fn,max_steps=10000000)

    def beamsearch(topk = 1):
        tokenizer = tokenization(DICT_PATH,DictSize=100000)
        source_file = queue_reader("NEWS", DATA_PATH )

        def _g():
            for source in source_file:
                value = source.split('#')
                source  = value[2].split(' ')
                title = value[0].split(' ')
                source = tokenizer.padding(tokenizer.tokenize(source),1000)
                title = tokenizer.padding(tokenizer.tokenize(title),100)
                yield  source,title
        g = _g()

        model_fn = build_model_fn(lr = 0.1,d_model=D_MODEL,input_vocab_size=VOCAB_SIZE,encoder_size=ENCODE_SIZE,encoder_layer_num=ENCODER_NUM,
                                  decoder_size=DECODER_SIZE,decoder_layer_num = DECODER_NUM)
        estimator = tf.estimator.Estimator(model_fn, model_dir=MODEL_PATH, )
        predictor = NewsPredictor(estimator,topk)
        bs = S2SBeamSearcher(dataset=g,tokenizer = tokenizer,topk=topk,context_len=10,predictor=predictor,max_count=100)
        bs.do_search_mt(100,estimator)
        bs.report('s2s_lstm')


    train()
    beamsearch(1)