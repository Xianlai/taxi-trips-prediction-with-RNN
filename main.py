#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script implements the LSTM model used to learn the spatio-temporal 
pattern of taxi trips.

Author: Xian Lai
Date: Apr 14, 2018
"""
from ops import *
from helpers import *
import tensorflow as tf
from rnn import RNN
from cell import *


def train(param, record_states=True, record_file=None, test=True):
    """
    """
    # prepare training and testing datasets
    ds_train = Dataset(
        train_features, train_targets,
        batch_size=param['batch_size'], 
        sequence_length=param['seq_len']
    )
    ds_test = Dataset(
        test_features, test_targets,
        batch_size=1, 
        sequence_length=param['seq_len']
    )
    seeds = list(range(param['n_epoch']))

    # prepare cell
    if param['cell'] == "VanillaCell":
        cell = ComposedCell(
            [VanillaCell(
                state_dim, 
                activation=param['activation'],
                use_encoder=param['use_encoder'], 
                use_decoder=param['use_decoder'],
                name="VanillaRNN_layer_"+str(i),
            ) for i, state_dim in enumerate(param['layer_sizes'])], 
            output_dim=ds_train.output_size
        )
    elif param['cell'] == "LSTMCell":
        cell = ComposedCell(
            [LSTMCell(
                state_dim,
                activation=param['activation'],
                use_encoder=param['use_encoder'], 
                use_decoder=param['use_decoder'],
                use_peephole=param['use_peephole'],
                name="BasicLSTM_layer_"+str(i),
            ) for i, state_dim in enumerate(param['layer_sizes'])],
            output_dim=ds_train.output_size
        )
    elif param['cell'] == "NTMCell":
        cell = ComposedCell(
            [NTMCell(
                state_dim,
                activation=param['activation'],
                use_encoder=param['use_encoder'], 
                use_decoder=param['use_decoder'],
                use_peephole=param['use_peephole'],
                n_write_head=param['n_write_head'],
                n_read_head=param['n_read_head'],
                controller=param['controller'],
                name="NTM_layer_"+str(i)
            ) for i, state_dim in enumerate(param['layer_sizes'])],
            output_dim=ds_train.output_size
        )

    # prepare chain
    chain = RNN(
        cell, ds_train, ds_test, n_epoch=param['n_epoch'], loss_fn=param['loss_fn'],
    )

    # fit the chain
    with tf.Session() as sess:
        states_train, outputs_train = chain.train(
            sess, ckpt_file, record_states=record_states, seeds=seeds
        )
        # variables_array = sess.run(get_variables())
        # pprint(get_variables())

    # test the chain
    if test:
        with tf.Session() as sess:
            states_test, results_test = chain.test(
                sess, ckpt_file, record_states=record_states, seeds=seeds
            )
            # variables_array = sess.run(get_variables())
            # pprint(get_variables())
            losses = np.array([x[2] for x in results_test])
            print("\nTest with 100 sequences,\n\tmean:%.4f\n\tstd:%.4f\n" % 
                (losses.mean(), losses.std()))
    else:
        states_test, results_test = (), ()

    # save the results and stats
    if record_file is not None:
        with open(data_path + record_file, "wb") as f:
            pickle.dump(
                (states_train, outputs_train, states_test, results_test), f
            )
        print("\nRecording saved to %s\n" %("../_data/recordings/" + file))

    print("THE END")


######################## model parameter setting #############################

if __name__ == '__main__':

    # cells  = ["VanillaCell", "LSTMCell", "NTMCell"] 
    # layers = [
    #     [25], [50], [100], [200], [400], [800], [1600], [3200],
    #     [100, 100], [100, 100, 100]
    # ]
    cells  = ["NTMCell"] 
    layers = [[100]]

    for i in range(1):
        print("--------:", i)
        for cell in cells:
            for l in layers:
                tf.reset_default_graph()
                param = dict(
                    batch_size   = 20,
                    seq_len      = 24,
                    layer_sizes  = l,
                    n_epoch      = 300,
                    activation   = tanh,
                    use_encoder  = True,
                    use_decoder  = True,
                    use_peephole = False,
                    loss_fn      = loss_l2,
                    n_write_head = 1,
                    n_read_head  = 1,
                    controller   = LSTMCell,
                    cell         = cell
                )
                print("model parameters:")
                pprint(param)
                file = "Recordings_%s_%s_20180609.pkl" % (
                    param['cell'], str(param['layer_sizes'])
                )
                train(param, record_states=True, record_file=file, test=True)


##############################################################################





