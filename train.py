# coding=utf-8

from itertools import product
from calculate import calculate
import os


def train(model, sess, saver, epochs, batch_size, data_train, id2word, id2tag):
  """
  模型训练
  :param model: model的一些信息
  :param sess: TensorFlow会话内容
  :param saver: TensorFlow保存器
  :param epochs: 所有训练数据forwward+bachword后更新参数的次数
  :param batch_size: 一次iteration选取的样本数量
  :param data_train: 输入样本
  :param id2word:    给每个word映射了一个id, 通过id 查找word
  :param id2tag:     给每个word映射了一个tag, 通过id 查找tag
  :return:
  """
  batch_num = int(data_train.y.shape[0] / batch_size)  # 计算一次iteration需要多少轮
  for epoch in range(epochs):
    for batch in range(batch_num):
      x_batch, y_batch = data_train.next_batch(batch_size)
      feed_dict = {model.input_data: x_batch, model.labels: y_batch}
      predict_train, _ = sess.run([model.viterbi_sequence, model.train_op], feed_dict=feed_dict)
      acc = 0  # 累计
      if batch % 100 == 0:
        for i, j in product(range(len(y_batch)), range(len(y_batch[0]))):
          if y_batch[i][j] == predict_train[i][j]:
            acc += 1
        recall_result = acc*1.0 / (len(y_batch)  * len(y_batch[0]))
        print("召回率：",recall_result )
    # 保存结果
    save_path_name = os.path.join(os.getcwd(), "model","model" + str(epoch) + ".ckpt")
    print(save_path_name)
    if epoch % 5 == 0:
      saver.save(sess, save_path_name)
      print("in epoch %d" % epoch, "model is saved")
      entity_result = [] # 应该是实体结果
      entity_all = [] # 和上面一样， 功能未知
      for batch in range(batch_num):
        x_batch, y_batch = data_train.next_batch(batch_size)
        feed_dict = {model.input_data: x_batch, model.labels: y_batch}
        predict_train = sess.run([model.viterbi_sequence], feed_dict)
        predict_train = predict_train[0] # 未知
        entity_result = calculate(x_batch, predict_train, id2word, id2tag, entity_result)
        entity_all = calculate(x_batch, y_train, id2word, id2tag, entity_all)
      and_set = set(entity_result) & set(entity_all)
      if len(and_set) > 0:
        precision_ratio = len(and_set) / len(entity_result)
        recall_ratio = len(and_set) / len(entity_all)
        f_value = 2 * precision_ratio * recall_ratio / (precision_ratio + recall_ratio)
        print("precision_ratio is :", precision_ratio)
        print("recall_ratio is : ",recall_ratio )
        print("f value is :", f_value)
      else:
        print("something wrong! no and set!!")