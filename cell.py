#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script implements different RNN cells and corresponding functions.

Author: Xian Lai
Date: Apr. 14, 2018
"""

from helpers import *
from ops import *

class _BasicCell():

    """
    This class implements the basic attributes and methods of rnn cell object.
    Because the terminologies used in RNN is confusing, for this project, I 
    define the terms as follow:
    """

    def __init__(self, state_dim, output_dim=None, initializer=norm_init, 
                 activation=tanh, dtype=tf.float32, use_encoder=True, 
                 use_decoder=True, name="RNNCell"):
        """
        Args:
            state_dim (int): The dimensionality of cell state, both long and 
                short term state.
            output_dim (int, default=None): 
                The dimensionality of prediction. If not set(None), it will be
                assumed same as state_dim. If set, the short term memory will
                be linear projected to this shape.
            initializer (tf.initializers default=random_normal): 
                the initializer for weights.
            activation (tf.activations) [optional, default=tf.tahn]: 
                the activation function for input gate and output gate.
            dtype (tf.dtype, default=tf.float32): dtype of state tensors
            use_encoder (bool) [optional, default=True]:
                whether to linearly project the input to state
            use_decoder (bool) [optional, default=True]:
                whether to linearly project the state to output
            name (str): name of this cell
        """
        self.state_dim    = state_dim
        self.output_dim   = output_dim if output_dim else state_dim
        self._initializer = initializer
        self._activation  = activation
        self._use_encoder = use_encoder
        self._use_decoder = use_decoder
        
        self.name  = name
        self.dtype = dtype


    def encode(self, input_, act_fn=identity_fn, name="encode"):
        """ linearly map the input from input_dim to state_dim.
        Args:
            input (tensor): input of current time step
            act_fn (function, default=identity_fn): 
                activation function applied after linear mapping
            name (str): variable scope of this op
        Returns:
            input (tensor): encoded input
        """
        inpt_dim = input_.shape[1]
        if (not self._use_encoder) and (inpt_dim != self.state_dim):
            raise Exception("Input dim must equal to state dim " + \
                            "when input encoding is disabled.")
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if self._use_encoder:
                input_, weights = linear(
                    input_, self.state_dim, name="input_encoding", 
                    return_weights=True
                )

        return act_fn(input_), weights


    def decode(self, st_state, name="decode"):
        """ linearly map the short term memory from state_dim to output_dim.
        Args:
            st_state (tensor): short term memory of current time step
            act_fn (function, default=identity_fn): 
                activation function applied after linear mapping
            name (str): variable scope of this op
        Returns:
            output_ (tensor): decoded input
        """
        if (not self._use_decoder) and (self.output_dim != self.state_dim):
            raise Exception("Output dim must equal to state dim " + \
                            "when output decoding is disabled.")
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if self._use_decoder:
                    output_ = self._activation(
                        linear(st_state, self.output_dim, "output_decoding")
                    )
            else:
                output_ = self._activation(st_state)
                
        return output_

    def init_state(self, batch_size):
        """ create an initial state for RNN chain
        Args:
            batch_size (int): the batch size of input data
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            zeros = tf.zeros(
                shape=[batch_size, self.state_dim],
                name="init_zeros", dtype=self.dtype
            )
            
        return {'lt_state': zeros, 'st_state': zeros}


class VanillaCell(_BasicCell):

    """
    This class implements the vanilla RNN cell in which cell state is modified
    by first concatenate previous state with current input and then linearly
    project to cell state size.
    """

    def __init__(self, state_dim, output_dim=None, initializer=norm_init, 
                 activation=tanh, dtype=tf.float32, use_encoder=True,  
                 use_decoder=True, name="VanillaCell"):
        """
        The state recorded:
            lt_state (batch_size, state_dim): cell state
            lt_state_o (batch_size, state_dim): 
                the ratio of cell state from last state
            lt_state_i (batch_size, state_dim): 
                the ratio of cell state from current input 
        """
        super().__init__(
            state_dim, output_dim=output_dim, initializer=initializer, 
            activation=activation, name=name, use_encoder=use_encoder,
            dtype=dtype, use_decoder=use_decoder
        )

    def __call__(self, states, input_):
        """ run one time step
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            input_, encoding_weights = self.encode(input_, act_fn=tanh)
            
            # merge flow of input and previous state and project to state size
            states['input'] = input_
            states['lt_state_prev'] = states['lt_state']
            xs = [states['lt_state'], input_]
            state = gate(xs, self.state_dim, tanh, name="update")
            output_ = self.decode(state)

            states['lt_state'] = state
            # states['encoding_weights'] = encoding_weights

        return states, output_


class LSTMCell(_BasicCell):

    """
    This class implements the Long Short Term Memory cell in which the modifi-
    cation of cell state is constrainted by gates. Besides the normal cell 
    state, it also produces a local memory(short term memory) used as one of 
    inputs to next time step.
    """

    def __init__(self, state_dim, output_dim=None, initializer=norm_init, 
                 activation=tanh, dtype=tf.float32, use_peephole=False,
                 use_encoder=True, use_decoder=True, name="LSTMCell_basic"):
        """
        Args:
            use_peepholes (bool, default=False): 
                whether allow gates seeing the cell state.

        The states of basic LSTM cell:
            lt_state (batch_size, state_dim): long term cell state
            st_state (batch_size, state_dim): short term cell state
            e_state (batch_size, state_dim): 
                intermediate long term state after forgot gate
            e_gate (batch_size, state_dim): 
                forget gate controlling delete information from long term state
            a_gate (batch_size, state_dim): 
                add gate controlling add information to long term state
            j_input (batch_size, state_dim): 
                new information to be added to long term state
            o_gate (batch_size, state_dim): 
                output gate controlling output from long term to short term state
        """
        super().__init__(
            state_dim, output_dim=output_dim, initializer=initializer, 
            activation=activation, name=name, use_encoder=use_encoder,
            dtype=dtype, use_decoder=use_decoder
        )
        self._use_peephole = use_peephole


    def __call__(self, states, input_):
        """ run one time step
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # inputs with shape [batch_size, input_size]
            input_, _ = self.encode(input_, act_fn=tanh)
            states['input'], states['st_state_prev'] = input_, states['st_state']

            # if use peepholes: f, i-gate has access to prev long term memory
            s_prev = [states['st_state'], input_]
            l_prev = s_prev.append(states['lt_state']) if self._use_peephole else s_prev

            # gates
            e_g = gate(l_prev, self.state_dim, sigmoid, name="erase_gate")
            a_g = gate(l_prev, self.state_dim, sigmoid, name="add_gate")
            o_g = gate(s_prev, self.state_dim, sigmoid, name="output_gate")
            j_g = gate(s_prev, self.state_dim, tanh, name="new_input")

            # states
            e_s = tf.multiply(states['lt_state'], e_g, name="erase_state")
            a_s = tf.add(e_s, a_g * j_g, name="add_state")
            o_s = tf.multiply(tanh(a_s), o_g, name="output_state")
            output_ = self.decode(o_s)
            
            states['e_gate'], states['e_state'] = e_g, e_s
            states['a_gate'], states['lt_state'] = a_g, a_s
            states['o_gate'], states['st_state'] = o_g, o_s
            states['j_input'] = j_g

        return states, output_


class Memory():

    """
    This class implements the Memory component of Neural Turing Machine cell
    served as a temporary space to save information.
    """
    def __init__(self, memory_size, memory_dim, shift_range=1, 
                 n_write_head=1, n_read_head=1, dtype=tf.float32, 
                 name="Memory"):
        """
        Args:
            memory_size (int, default=128): the number of memory slots
            memory_dim (int, default=20)  : the size of memory slot
            shift_range (int, default=1)  : the range allowed for shifting heads
            n_write_head (int, default=1) : number of write heads
            n_read_head (int, default=1)  : number of read heads

        state:
            memory: tensor with shape:(batch_size, memory_size, memory_dim)
            r_weights: tensor with shape:(batch_size, memory_size)
            w_weights: tensor with shape:(batch_size, memory_size)
            memory_prev
            read: tensor with shape:(batch_size, memory_dim)
            erase
            write
            r_weight_c
            r_weight_g
            r_wgt_sml
            r_wgt_beta
            w_weight_c
            w_weight_g
            w_wgt_sml
            w_wgt_beta
        """
        self.memory_size  = memory_size
        self.memory_dim   = memory_dim
        self.shift_range  = 2 * shift_range + 1 
        self.n_write_head = n_write_head
        self.n_read_head  = n_read_head
        self.name  = name
        self.dtype = dtype


    def read(self, states, input_):
        """ read information from memory based on current input and previous 
        short term memory.

        Args:
            states: states of cell
            input_: input of cell
        """
        # memory has shape (memory_size, memory_dim), shared by the whole batch
        # read weight for each head has shape (batch_size, memory_size)
        # read for each head has shape (batch_size, state_dim) after decode
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # Reading: read_t <- r_weight X memory_added_{t-1}
            with tf.variable_scope("read"):
                # encoded input_ has shape (batch_size, memory_dim)
                flows   = tf.concat([states['st_state'], input_], axis=1)
                encoded = self.encode(flows, name="encode_key")
                sml_coss, sml_betas, sml_shps, smls = [], [], [], []
                wgt_gs, wgt_gammas, wgt_itpls, wgt_convs, wgt_shps = [], [], [], [], []
                r_weights_prevs, r_weights, reads = [], [], []
                for i, r_wgt_prev in enumerate(states['r_weights']):
                    with tf.variable_scope("head_%d" % i):
                        r_smls, r_wgts = self.focus(encoded, states['memory'], r_wgt_prev, True)
                        sml_cos, sml_beta, sml_shp, sml = r_smls
                        wgt_g, wgt_gamma, wgt_itpl, wgt_conv, wgt_shp, r_wgt = r_wgts
                        read = r_wgt @ states['memory']  # (batch_size, memory_dim)
                        read = self.decode(read, states['lt_state'].shape[-1], 
                            name="decode_memory"
                        )  # (batch_size, state_dim)

                        sml_coss.append(sml_cos)
                        sml_betas.append(sml_beta)
                        sml_shps.append(sml_shp)
                        smls.append(sml)
                        wgt_gs.append(wgt_g)
                        wgt_gammas.append(wgt_gamma)
                        wgt_itpls.append(wgt_itpl)
                        wgt_convs.append(wgt_conv)
                        wgt_shps.append(wgt_shp)
                        r_weights_prevs.append(r_wgt_prev)
                        r_weights.append(r_wgt)
                        reads.append(read)

                read = tanh(tf.add_n(reads, name="merge_heads"))  # (batch_size, state_dim)

            states['memory_read'] = read
            states['memory_prev'] = states['memory']
            states['r_weights_prev'] = r_weights_prevs
            states['r_weights']   = r_weights
            states['r_sml_cos']  = sml_coss
            states['r_sml_beta'] = sml_betas
            states['r_sml_shp'] = sml_shps
            states['r_sml']      = smls
            states['r_wgt_g']     = wgt_gs
            states['r_wgt_gamma'] = wgt_gammas
            states['r_wgt_itpl']  = wgt_itpls
            states['r_wgt_conv']  = wgt_convs
            states['r_wgt_shp']   = wgt_shps

        return states


    def write(self, states):
        """ Write information to memory based on current long term memory.
        {b: batch_size, n: memory_size, m: memory_dim}
        Args:
            states: states of cell
        """
        
        # memory has shape (n, m)
        # write weight has shape (b, n)
        # both erase & write has shape (n, m)
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # Writing:
            # memory_erased_t <- memory_added_{t-1} * [I - w_weight_t.T X e_t]
            # memory_added_t <- memory_erased_t + w_weight_t.T X a_t
            with tf.variable_scope("write"):
                sml_coss, sml_betas, sml_shps, smls = [], [], [], []
                wgt_gs, wgt_gammas, wgt_itpls, wgt_convs, wgt_shps = [], [], [], [], []
                w_weights_prevs, w_weights, erases, writes = [], [], [], []
                es, as_, Is, ws = [], [], [], []
                # encoded input_ has shape (b, m)
                encoded = self.encode(states['st_state'], name="encode_key")
                for i, w_wgt_prev in enumerate(states['w_weights']):
                    with tf.variable_scope("head_%d" % i):
                        w_smls, w_wgts = self.focus(encoded, states['memory'], w_wgt_prev, False)
                        sml_cos, sml_beta, sml_shp, sml = w_smls
                        wgt_g, wgt_gamma, wgt_itpl, wgt_conv, wgt_shp, w_wgt = w_wgts


                        I = tf.ones(states['memory'].shape, self.dtype) # (n, m)
                        w = tf.transpose(w_wgt) # (n, b)
                        e = sigmoid(encoded, name='erase')  # (b, m)
                        a = tanh(encoded, name='add')  # (b, m)
                        
                        erase_ = w @ e  # (n, m)
                        erases.append(I - erase_)
                        write = w @ a  # (n, m)
                        writes.append(write)

                        # -------------- recording --------------
                        as_.append(w_wgt)
                        es.append(e)
                        Is.append(erase_)

                        sml_coss.append(sml_cos)
                        sml_betas.append(sml_beta)
                        sml_shps.append(sml_shp)
                        smls.append(sml)

                        wgt_gs.append(wgt_g)
                        wgt_gammas.append(wgt_gamma)
                        wgt_itpls.append(wgt_itpl)
                        wgt_convs.append(wgt_conv)
                        wgt_shps.append(wgt_shp)

                        w_weights_prevs.append(w_wgt_prev)
                        w_weights.append(w_wgt)
                        

                erase = reduce(lambda x, y: x * y, erases)
                write = tf.add_n(writes)
                memory_new = tanh(states['memory'] * erase + write)

            states['memory']   = memory_new
            states['w_encoded'] = encoded
            states['es']   = es
            states['Is']   = Is
            states['ws']   = w
            states['erases']   = erases
            states['as']   = as_
            states['erase']    = erase
            states['write']    = write
            states['w_weights_prev'] = w_weights_prevs
            states['w_sml_cos']  = sml_coss
            states['w_sml_beta'] = sml_betas
            states['w_sml_shp']  = sml_shps
            states['w_sml']      = smls
            states['w_wgt_g']    = wgt_gs
            states['w_wgt_gamma'] = wgt_gammas
            states['w_wgt_itpl'] = wgt_itpls
            states['w_wgt_conv'] = wgt_convs
            states['w_wgt_shp']  = wgt_shps
            states['w_weights'] = w_weights

        return states


    def focus(self, encoded, memory_prev, weight_prev, is_read):
        """ generate weights with shape (batch_size, 1, memory_size) for either
        reading or writing using content and location focus.
        {b: batch_size, n: memory_size, m: memory_dim}

        Args:
            encoded (tensor with shape (b, m)): 
                encoded long term controller state for writing or encoded
                concatenated inputs for reading
            memory_prev (tensor with shape (n, m)):
                memory state before updating
            weight_prev (tensor with shape (b, n)):
                previous weights for either writing or reading
        Returns:
            wgt (tensor with shape (b, n)):
                weights for current reading or writing
        """
        with tf.variable_scope("focus", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("by_content"):
                # FOCUS BY CONTENT: 
                # - key vector: sml_k with shape (b, m)
                # - Cosine similarity: sml_cos with shape (b, n)
                # - similarities scaled: sml_scal with shape (b, n)
                # - positive key strength: sml_beta with shape (b, 1)
                # - similarities shapened: sml_shp with shape (b, n)
                # - final similarities: sml with shape (b, n)
                sml_k    = tanh(encoded, name="key_vec")
                sml_cos  = cos_similarity(sml_k, memory_prev)
                sml_scal = minmax_scale_tensor(sml_cos)
                sml_beta = softplus(linear(encoded, 1, name='beta')) + 1
                sml_shp  = tf.pow(sml_scal, sml_beta, name='sharpen')
                sml      = softmax(sml_shp * (20 if is_read else 70)) #---------------------------------#

            with tf.variable_scope("by_location"):
                # FOCUS BY LOCATION:
                # - interpolation gate: wgt_g with shape (b, 1)
                # - shift weighting: wgt_s with shape (b, shift_range)
                # - sharpening power: wgt_gamma with shape (b, 1)
                wgt_g = tf.sigmoid(linear(encoded, 1, name='g')) / 2
                wgt_s = softmax(linear(encoded, self.shift_range, name="s"))
                wgt_gamma = softplus(linear(encoded, 1, name='gamma')) + 1

                # interpolated weight: wgt_itpl with shape (b, n)
                # convoluted weight: wgt_conv with shape (b, n)
                # sharpened weight: wgt_shp with shape (b, n)
                # normalize term: norm with shape (b, 1)
                # normalized weight: wgt with shape (b, n)
                wgt_itpl = wgt_g * sml + (1 - wgt_g) * weight_prev
                wgt_conv = circular_convolute(wgt_itpl, wgt_s)
                wgt_shp  = tf.pow(wgt_conv, wgt_gamma)
                norm     = tf.reduce_sum(wgt_shp, axis=1, keepdims=True)
                wgt      = wgt_shp / norm

            smls = (sml_cos, sml_beta, sml_shp, sml)
            wgts = (wgt_g, wgt_s, wgt_itpl, wgt_conv, wgt_shp, wgt)

            return smls, wgts


    def encode(self, output_, name="encode"):
        """ linear map the output of controller with shape (batch_size, 
        state_dim) to shape (batch_size, memory_dim)
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if output_.shape[1] != self.memory_dim:
                encoded = tanh(
                    linear(output_, self.memory_dim, "linear_proj")
                )
            else:
                encoded = output_
        return encoded


    def decode(self, read, output_dim, name="encode"):
        """ linear map the read with shape (batch_size, memory_size, 
        memory_dim) to shape (batch_size, memory_size, state_dim)
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if output_dim != self.memory_dim:
                batch_size, m_dim = read.shape
                w = tf.get_variable(
                    "weights",
                    shape=[m_dim, output_dim],
                    dtype=self.dtype
                )
                b = tf.get_variable(
                    "bias",
                    shape=[output_dim],
                    dtype=self.dtype
                )
                return read @ w + b
            else:
                return read


    def init_state(self, batch_size, init_memory):
        """ Initialize the states of memory
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # init_weights = tf.transpose(tf.one_hot(
            #     [0]*batch_size, depth=self.memory_size, axis=0, 
            #     dtype=self.dtype, name="init_weights"
            # ))
            init_weights = tf.transpose(tf.random_shuffle(tf.one_hot(
                [0]*batch_size, depth=self.memory_size, axis=0, 
                dtype=self.dtype, name="init_weights"
            )))

            return {
                'memory'   :init_memory,
                'r_weights':[init_weights] * self.n_read_head,
                'w_weights':[init_weights] * self.n_write_head,
            }


class NTMCell():

    """
    This class implements the Neural Turing Machine cell which is an RNN cell
    (controller) that use a memory bank to temporaly store cell's long term 
    memory. After output gate, the short term memory is writen to the memory  
    and the information in memory before written is read as the input of short 
    term cell state input of next cell. 

    It has 2 main components: memory and controller. And its state is the 
    combination of state of memory and state of controller. 
    """
    def __init__(self, state_dim, output_dim=None, initializer=norm_init, 
                 activation=tanh, dtype=tf.float32, use_peephole=False, 
                 use_encoder=True, use_decoder=True, controller=LSTMCell,
                 memory_size=128, memory_dim=None, shift_range=2, n_write_head=1,
                 n_read_head=1, name="NTMCell"):
        """
        """
        self.controller = controller(
            state_dim, output_dim=output_dim, initializer=initializer, 
            activation=activation, dtype=dtype, use_peephole=use_peephole,
            use_encoder=use_encoder, use_decoder=use_decoder, 
            name="NTM_controller"
        )
        self.memory = Memory(
            memory_size=memory_size, memory_dim=memory_dim if memory_dim else state_dim, 
            shift_range=shift_range, n_write_head=n_write_head,
            n_read_head=n_read_head, name="NTM_memory",
        )
        self.name = name
        self.output_dim = output_dim


    def __call__(self, states, input_):
        """ run one time step.
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # inputs with shape [batch_size, input_size]
            input_, _ = self.controller.encode(input_, act_fn=tanh)
            states['encoded_input'] = input_
            states['prev_lt_state'] = states['lt_state']
            states['prev_st_state'] = states['st_state']
            # read from memory
            states = self.memory.read(states, input_)

            # if use peepholes: f, i-gate has access to prev long term memory
            s_prev = [states['st_state'], states['memory_read'], input_]
            # s_prev = [states['memory_read'], input_]
            l_prev = s_prev.append(states['lt_state']) if self.controller._use_peephole else s_prev

            # gates
            e_g = gate(l_prev, self.controller.state_dim, sigmoid, name="erase_gate")
            a_g = gate(l_prev, self.controller.state_dim, sigmoid, name="add_gate")
            o_g = gate(s_prev, self.controller.state_dim, sigmoid, name="output_gate")
            j_g = gate(s_prev, self.controller.state_dim, tanh, name="new_input")

            # states
            e_s = tf.multiply(states['lt_state'], e_g, name="erase_state")
            a_s = tf.add(e_s, a_g * j_g, name="add_state")
            o_s = tf.multiply(tanh(a_s), o_g, name="output_state")
            output_ = self.controller.decode(o_s)
            
            states['e_gate'], states['e_state'] = e_g, e_s
            states['a_gate'], states['lt_state'] = a_g, a_s
            states['o_gate'], states['st_state'] = o_g, o_s
            states['j_input'] = j_g

            # update memory
            states = self.memory.write(states)
            states['output'] = output_
        return states, output_


    def init_state(self, batch_size, init_memory):
        """ Initialize the cell state.
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE): 
            states = self.controller.init_state(batch_size)
            states.update(
                self.memory.init_state(batch_size, init_memory)
            )

            return states



class ComposedCell():

    """ A wrapper class to stack given cells.
    """
    def __init__(self, cells, output_dim, name="composed_cell"):
        """
        Args:
            cells (list): a list of RNN cells to be stacked.
            output_dim (int): the dimensionality of output
            name (str): the name of this cell.
        """
        self._cells = cells
        self.name = name
        if "NTM" in self._cells[-1].name:
            self._cells[-1].controller.output_dim = output_dim
        else:
            self._cells[-1].output_dim = output_dim


    def __call__(self, states_lst, input_):
        """ 
        Args:
            states_lst (list): a list of previous cell states in each layer.
            input_ (tensor): the input data of current time step

        Returns:
            new_states_lst (list): the list of current cell states  
            input_curr (tensor): the output of current cell
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            input_curr = input_
            new_states_lst = []
            for i, cell in enumerate(self._cells):
                state_curr = states_lst[i]
                new_states, input_curr = cell(state_curr, input_curr)
                new_states_lst.append(new_states)

        return new_states_lst, input_curr


    def init_state(self, batch_size, init_memories=[]):
        """ Create an initial state.

        Args:
            batch_size (int): the batch size of input data
            init_memories (list) [optional, default=[]]: the list of memory
                shape in each layer. If empty, no memories are used.
        Returns:
            states (list): the initial state of this RNN chain of all layers.
        """
        init_memories = list(init_memories)
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE): # level: composed_cell
            states = []
            for cell in self._cells:
                if "NTM" in cell.name:
                    state = cell.init_state(batch_size, init_memories.pop())
                else:
                    state = cell.init_state(batch_size)
                states.append(state)

            return states



