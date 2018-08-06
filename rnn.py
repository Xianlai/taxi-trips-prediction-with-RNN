#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script implements the RNN chain model.

Author: Xian Lai
Date: Mar. 03, 2018
"""

from helpers import *
from cell import *
from ops import *
from copy import deepcopy


class RNN(object):

    """ 
    This class implements the functions to build, train and test the 
    uni-directional RNN model with given RNN cell and datasets.
    """

    def __init__(self, cell, ds_train, ds_test, n_epoch=150, loss_fn="L1", 
                 name="RNN"):
        """
        Args:
            cell (RNNCell): an RNN cell instance.
            ds_train (array): the training dataset instance.
            ds_test (array): the testing dataset instance.
            n_epoch (int): the maximal iterations.
            loss_fn (function): Either loss_l1 or loss_l2 defined in ops.py. 
        """
        self.cell     = cell
        self.ds_train = ds_train
        self.ds_test  = ds_test
        self.loss_fn  = loss_fn
        self.n_epoch  = n_epoch
        self.name     = name

        self.input_dim  = ds_train.input_size
        self.output_dim = 1
        self.batch_size = ds_train.batch_size
        self.seq_len    = ds_train.seq_len
        self.dtype      = tf.float32
         

    def _build(self, batch_size, memory_shapes=[]):
        """ Build the graph of RNN model either in train mode or test mode.
        The only difference between train and test mode is the batch_size.

        Args:
            batch_size (int): the batch_size to define shapes of placeholders.
            memory_shapes (list, default=[]): the list of memory
                shape in each layer. If empty, no memories are used.
        """
        # build the initial state of chain
        self.m = [
            tf.placeholder(self.dtype, shape=shp, name='init_memory_'+str(i)) \
            for i, shp in enumerate(memory_shapes)
        ]
        states = self.cell.init_state(batch_size, init_memories=self.m)

        # build the inputs of chain
        self.x = tf.placeholder(
            self.dtype, name='inputs',
            shape=(self.seq_len, batch_size, self.input_dim)
        )
        inputs = tf.unstack(
            self.x, num=self.seq_len, axis=0, name="unstack_input_tensor"
        )

        # build the chain and save the intermediate states(top layer only)
        self.records = []
        for time_step in range(self.seq_len):
            states, self.predict = self.cell(states, inputs[time_step])
            self.records.append(states[-1].copy())

        # record memory of last seq_step as init_memory of next iteration
        if not (memory_shapes == []):
            self.memory = [state.copy()['memory'] for state in states]

        # build the evaluation of chain
        self.y = tf.placeholder(
            self.dtype, name='target',
            shape=(batch_size, self.output_dim)
        )
        # self.y = tf.Print(self.y, [])
        self.loss = self.loss_fn(self.y, self.predict)


    def train(self, sess, checkpoint, record_states=True, seeds=None):
        """ train the model to minimize the loss.

        Args:
            sess (tf.Session): the session used to run built graph
            checkpoint (str): the path to save the trained model
            record (bool, default=True): whether to record the states
            sees (float, default=None): random seed to generate input data batch

        Returns:
            - input_iters (list): the input data for all iterations, each 
                element is an array with shape (seq_len, batch_size, input_dim)
                for one iteration.
            - record_iters (list): the recorded states for all iterations, each
                element is a list with length seq_len, each element is a dict
                of state in one seq_step(top layer if composed cell).
            - output_iters (list): the (prediction, label, loss) for all 
                iterations, each prediction and label is an array with same
                shape as grids. each loss is a scalar for one iteration.
        """
        # get the memory shape and init memory as zeros of each layer
        memory_shapes, init_memories = [], []
        for cell in self.cell._cells:
            if "NTM" in cell.name:
                m_shape = (cell.memory.memory_size, cell.memory.memory_dim)
                memory_shapes.append(m_shape)
                init_memories.append(np.zeros(shape=m_shape, dtype=np.float64))

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # build graph for training
            self._build(self.batch_size, memory_shapes=memory_shapes)
            # optimizer = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.5, use_nesterov=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.9, beta2=0.9)
            train_op  = optimizer.minimize(self.loss)

            print("\nTraining begins....\n")
            sess.run(tf.global_variables_initializer())
            states_train, results_train = [], []
            for epoch in range(self.n_epoch):
                gc.collect(); time_ = timer()

                # prepare input sequence
                input_data, ranges = self.ds_train.next_batch(random_seed=seeds[epoch])
                target = self.ds_train.get_target(ranges=ranges)
                feed_dict = {k:v for k, v in zip(self.m, init_memories)}
                feed_dict.update({self.x: input_data, self.y: target})

                # run the graph
                sess.run(train_op, feed_dict=feed_dict)
                predict, loss = sess.run([self.predict, self.loss], feed_dict=feed_dict)

                if not memory_shapes == []:
                    init_memories = sess.run(self.memory, feed_dict=feed_dict)

                # if (not (memory_shapes == [])) and (epoch == self.n_epoch - 2):
                #     init_memories_2 = sess.run(self.memory, feed_dict=feed_dict)

                # print out the loss of each iteration and transform loss back
                # to original unit and scale for intuitive interpretation
                if self.loss_fn == loss_l2: loss = math.sqrt(loss)
                loss *= self.ds_train.targets_rng
                results_train.append((sparsify(predict[0]), sparsify(target[0]), loss))
                print("\titeration: {:<8} Cost time:{:<15} Batch loss:{:<7.4f}".\
                    format(epoch, elapsed(timer() - time_), loss))
                print("train results: (%3f, %3f, %3f)" %(predict[0], target[0], loss))
                if record_states:
                    states_seq = sess.run(self.records, feed_dict=feed_dict)
                    # init_memories = [states_seq[-1]['memory']] # states_seq: iter, ts, ['memory']
                    states_seq = [take_1st(states) for states in states_seq]
                    states_train.append(states_seq)

        # save memory shapes and states to pkl file
        if not (memory_shapes == []):
            with open(data_path + self.name + "memory.pkl", "wb") as f: 
                pickle.dump((memory_shapes, init_memories), f)
        # if not (memory_shapes == []):
        #     with open(data_path + self.name + "memory_2.pkl", "wb") as f: 
        #         pickle.dump((memory_shapes, init_memories_2), f)

        # save model to checkpoint dir and summaries to log dir
        tf.train.Saver().save(sess, checkpoint)
        print("\nTraining Finished. \nModel saved to {}".format(checkpoint))
        # writer = tf.summary.FileWriter(log_path, sess.graph)

        return states_train, results_train


    def test(self, sess, checkpoint, record_states=False, seeds=None):
        """ test the trained model for 100 sequence and return the mean and std.

        Args:
            sess (tf.Session): the session used to run built graph
            checkpoint (str): the path to save the trained model
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # load the learned model memory if exist
            try:
                with open(data_path + self.name + "memory.pkl", "rb") as f: 
                    memory_shapes, init_memories = pickle.load(f)
            except FileNotFoundError:
                memory_shapes, init_memories = [], []

            # build model with test mode and load the learned variable
            self._build(self.ds_test.batch_size, memory_shapes=memory_shapes)
            tf.train.Saver().restore(sess, checkpoint)

            states_test, results_test = [], []
            for epoch in range(self.n_epoch):
                # prepare the feed_dict
                input_data, ranges = self.ds_test.next_batch(random_seed=seeds[epoch])
                target = self.ds_test.get_target(ranges=ranges)
                feed_dict = {k:v for k, v in zip(self.m, init_memories)}
                feed_dict.update({self.x: input_data, self.y: target})

                # run the graph
                predict, loss = sess.run(
                    [self.predict, self.loss], feed_dict=feed_dict
                )

                if self.loss_fn == loss_l2: loss = math.sqrt(loss)
                loss *= self.ds_test.targets_rng
                print("test results: (%3f, %3f, %3f)" %(predict[0], target[0], loss))
                results_test.append(
                    (sparsify(predict[0]), sparsify(target[0]), loss)
                )

                if record_states:
                    states_seq = sess.run(self.records, feed_dict=feed_dict)
                    states_seq = [take_1st(states) for states in states_seq]
                    states_test.append(states_seq)

        return states_test, results_test


            





