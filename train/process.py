import os,time,logging,sys
from data_util import data_pipe
from model import Models
from meta.Meta import  Meta
import rouge
import tensorflow as tf
def run_train_task(**kwargs):
    TaskName = kwargs['TaskName']
    epochSize = kwargs['EpochSize']
    logInterval = kwargs['LogInterval']
    dataPipe = dataPipe.DataPipe(**kwargs)
    model = Models.unionGenerator(**kwargs)
    inputPipe = dataPipe.read_TFRecord(model.BatchSize)
    ops = model.build_model_pipe(mode='train',input = inputPipe)
    initOp = tf.initialize_all_variables()

    epoch = kwargs['Epoch']
    if 'CKP_DIR' not in kwargs:
        kwargs['CKP_DIR'] = 'checkpoint_'+TaskName+'/'

    if 'SUMMARY_DIR' not in kwargs:
        kwargs['SUMMARY_DIR'] = 'summary_' + TaskName + '/'

    checkpoint_dir = os.path.abspath(kwargs['CKP_DIR'])  # meta
    summary_dir = os.path.abspath(kwargs['SUMMARY_DIR'])  # meta
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    # data_meta = kwargs['DataMeta']  # meta
    # 模型搭建
    # with tf.device('/cpu:0'):
    # with tf.device('/device:GPU:0'):

    # 训练过程
    saver = tf.train.Saver()
    config = tf.ConfigProto(
        # log_device_placement=True

    )
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # 训练配置，包括参数初始化以及读取检查点

        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        sess.graph.finalize()
        train_writer = tf.summary.FileWriter(summary_dir, sess.graph)
        start_epoch = 0
        global_step = 0
        if checkpoint:
            saver.restore(sess, checkpoint)
            print('[INFO] 从上一次的检查点:\t%s开始继续训练任务' % checkpoint)
            start_epoch += int(checkpoint.split('-')[-1])
            global_step += int(checkpoint.split('-')[-2])
        else:
            sess.run(initOp)
        start_time = time.time()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        # 开始训练

        for i in range(start_epoch, epoch):
            try:
                batch_count = 0
                for b in range(epochSize):
                    try:
                        last_time = time.time()

                        loss,_,merge,lr = sess.run([ops['loss'],ops['train'],ops['merge'],ops['LearningRate']],feed_dict={
                            ops['GlobalStep']:global_step
                        })
                        cur_time = time.time()
                        time_cost = cur_time - last_time
                        total_cost = cur_time - start_time
                        if global_step % max(logInterval,1) == 0:
                            train_writer.add_summary(merge, global_step)
                            # logger.write_log([global_step/10,loss,total_cost])
                        print('[INFO] Batch %d 训练结果：LOSS=%.2f  学习率：%.2e用时: %.2f 共计用时 %.2f' % (
                        batch_count, loss,lr,time_cost, total_cost))

                        # print('[INFO] Batch %d'%batch_count)
                        # matplotlib 实现可视化loss
                        batch_count += 1
                        global_step += 1

                    except Exception as e:
                        logging.exception(e)
                        print("[INFO] 因为程序错误停止训练，开始保存模型")
                        saver.save(sess, os.path.join(checkpoint_dir, kwargs['TaskName'] + '_summary-' + str(global_step)),
                                   global_step=i)

                print("[INFO] Epoch %d 结束，现在开始保存模型..." % i)
                saver.save(sess, os.path.join(checkpoint_dir, kwargs['TaskName'] + '_summary-' + str(global_step)),
                           global_step=i)
            except KeyboardInterrupt:
                print("[INFO] 强行停止训练，开始保存模型")
                saver.save(sess, os.path.join(checkpoint_dir, kwargs['TaskName'] + '_summary-' + str(global_step)),
                           global_step=i)
                break
        coord.request_stop()
        coord.join(threads)

def run_eval_task_selmap(**kwargs):

    # 从原文中进行选择的版本
    TaskName = kwargs['TaskName']
    evalCaseNum = kwargs['EvalCaseNum']

    dataPipe = dataPipe.DataPipe(**kwargs)
    model = Models.unionGenerator(**kwargs)
    dataProvider = dataPipe.pipe_data_for_eval(**kwargs)

    ops = model.build_model_pipe(mode='infer',input = None)
    probThresh = kwargs['ProbThresh']
    if 'CKP_DIR' not in kwargs:
        kwargs['CKP_DIR'] = 'checkpoint_'+TaskName+'/'

    if 'SUMMARY_DIR' not in kwargs:
        kwargs['SUMMARY_DIR'] = 'summary_' + TaskName + '/'

    checkpoint_dir = os.path.abspath(kwargs['CKP_DIR'])  # meta
    # summary_dir = os.path.abspath(kwargs['SUMMARY_DIR'])  # meta
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    # 模型搭建
    # 训练过程
    saver = tf.train.Saver()
    config = tf.ConfigProto(
        # log_device_placement=True

    )
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # 训练配置，包括参数初始化以及读取检查点

        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        # train_writer = tf.summary.FileWriter(summary_dir, sess.graph)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print('[INFO] 从上一次的检查点:\t%s开始继续训练任务' % checkpoint)
        else:
            print('[ERROR] 没有找到任何匹配的checkpoint文件')
        sess.graph.finalize()
        start_time = time.time()
        # 开始训练
        for i in range(evalCaseNum):
            try:
                batch_count = 0
                last_time = time.time()

                line,flagSeq,topicSeq,refMap,refVector = next(dataProvider)

                for v in range(len(line)):

                     PW,PT,PF,PS,SV = sess.run([ops['wordRes'],
                                             ops['topicRes'],
                                             ops['flagRes'],
                                             ops['selRes'],
                                             ops['selMap']],
                                            feed_dict={
                                                ops['keyWordVector'] : refVector,
                                                ops['wordVector'] : 0,
                                                ops['topicSeq'] : topicSeq,
                                                ops['flagSeq'] : flagSeq
                                            })
                     if PS>probThresh:
                        pass
                        # PWTF =

                cur_time = time.time()
                time_cost = cur_time - last_time
                total_cost = cur_time - start_time
                print('[INFO] Sample %d 验证结果：Pre=%.2f  用时: %.2f 共计用时 %.2f' % (
                        i, 0, time_cost, total_cost))

            except KeyboardInterrupt:
                print("[INFO] 强行停止验证 开始保存结果")

                break


def run_eval_task_gen(**kwargs):
    # 直接进行生成的版本
    TaskName = kwargs['TaskName']
    evalCaseNum = kwargs['EvalCaseNum']

    dataPipe = dataPipe.DataPipe(**kwargs)
    model = Models.unionGenerator(**kwargs)
    dataProvider = dataPipe.pipe_data_for_eval(**kwargs)

    ops = model.build_model_pipe(mode='infer', input=None)
    if 'CKP_DIR' not in kwargs:
        kwargs['CKP_DIR'] = 'checkpoint_' + TaskName + '/'

    if 'SUMMARY_DIR' not in kwargs:
        kwargs['SUMMARY_DIR'] = 'summary_' + TaskName + '/'

    checkpoint_dir = os.path.abspath(kwargs['CKP_DIR'])  # meta
    # summary_dir = os.path.abspath(kwargs['SUMMARY_DIR'])  # meta
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    # 模型搭建
    # 训练过程
    saver = tf.train.Saver()
    config = tf.ConfigProto(
        # log_device_placement=True

    )
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # 训练配置，包括参数初始化以及读取检查点

        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        # train_writer = tf.summary.FileWriter(summary_dir, sess.graph)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print('[INFO] 从上一次的检查点:\t%s开始进行验证' % checkpoint)
        else:
            print('[ERROR] 没有找到任何匹配的checkpoint文件')
        sess.graph.finalize()
        start_time = time.time()
        # 开始训练
        hyps = []
        refs = []
        for i in range(evalCaseNum):
            try:
                batch_count = 0
                last_time = time.time()

                line, flag, topic, refMap, refVector,refWords = next(dataProvider)

                wordSeq,topicSeq,flagSeq = dataPipe.get_input_data()
                wordRes = []

                for v in range(len(line)):

                    PW, PT, PF = sess.run([ops['wordRes'],
                                                   ops['topicRes'],
                                                   ops['flagRes']],
                                                  feed_dict={
                                                      ops['keyWordVector']: refVector,
                                                      ops['wordVector']: wordSeq,
                                                      ops['topicSeq']: topicSeq,
                                                      ops['flagSeq']: flagSeq
                                                  })
                    newWord,newFlag,newTopic = dataPipe.get_prob_result(PW,PF,PT)
                    wordSeq, topicSeq, flagSeq = dataPipe.get_input_data(wordSeq, topicSeq, flagSeq, newWord)
                    wordRes.append(newWord)
                wordRes = ''.join(wordRes)
                wordRes = ' '.join(list(wordRes))
                print(wordRes)
                refWords = ' '.join(refWords)
                print(refWords)
                cur_time = time.time()
                time_cost = cur_time - last_time
                total_cost = cur_time - start_time

                line = ''.join(line)
                line = ' '.join(list(line))
                hyps.append(wordRes)
                refs.append(line)
                print('[INFO] Sample %d 验证结果：Pre=%.2f  用时: %.2f 共计用时 %.2f' % (
                    i, 0, time_cost, total_cost))

            except KeyboardInterrupt:
                print("[INFO] 强行停止验证 开始保存结果")

                break
        res = rouge.Rouge().get_scores(hyps=hyps, refs=refs,avg=True)
        print(res)

