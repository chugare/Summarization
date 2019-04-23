import Tool
import LDA
import os,time,logging,sys
import DataPipe,Models

import tensorflow as tf
def run_train_task(**kwargs):
    TaskName = kwargs['TaskName']
    epochSize = kwargs['EpochSize']
    dataPipe = DataPipe.DataPipe(**kwargs)
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
        start_time = time.time()
        sess.run(initOp)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        # 开始训练
        for i in range(start_epoch, epoch):
            try:
                batch_count = 0
                for b in range(epochSize):
                    try:
                        last_time = time.time()

                        loss,_,merge = sess.run([ops['loss'],ops['train'],ops['merge']])
                        cur_time = time.time()
                        time_cost = cur_time - last_time
                        total_cost = cur_time - start_time
                        if global_step % 100 == 0:
                            train_writer.add_summary(merge, global_step)
                            # logger.write_log([global_step/10,loss,total_cost])
                            print('[INFO] Batch %d 训练结果：LOSS=%.2f  用时: %.2f 共计用时 %.2f' % (
                        batch_count, loss, time_cost, total_cost))

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
def run_eval_task(**kwargs):
    TaskName = kwargs['TaskName']
    evalCaseNum = kwargs['EvalCaseNum']

    evalProvider = DataPipe.EvalProvider(**kwargs)
    model = Models.unionGenerator(**kwargs)
    dataProvider = evalProvider.read_data_for_eval(**kwargs)

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
                        batch_count, 0, time_cost, total_cost))

                batch_count += 1
            except KeyboardInterrupt:
                print("[INFO] 强行停止验证 开始保存结果")

                break

LDA_TRAIN={
    'TaskName':'LDA_TRAIN',
    'SOURCE_FILE':'data.txt',


}
if __name__ == '__main__':
    args = sys.argv
    run_train_task(TaskName='DP',
             Epoch=10,    # 训练的迭代次数
             EpochSize=100000,# 每一个迭代当中的数据量
             BatchSize=64,# 训练的批的大小
             ReadNum = int(args[1]) # 从词向量当中读取的单词的数量，-1表示全部读取，读取大量词向量需要消耗大量的时间
             )