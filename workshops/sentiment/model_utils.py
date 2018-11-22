import tensorflow as tf
import utils.io


def accuracy(labels, predictions, scope='acc'):
    with tf.variable_scope(scope):
        acc_value, update_op = tf.metrics.accuracy(labels=labels, predictions=predictions)
        vars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(vars)
        return acc_value, update_op, reset_op


def rnn_gru(inputs, units, sequence_length=None):
    cell = tf.nn.rnn_cell.GRUCell(num_units=units, activation=tf.nn.tanh, name="GRU_cell")
    _, last_states = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, sequence_length=sequence_length, dtype=tf.float32)
    return last_states


def rnn_vanilla(inputs, units, sequence_length=None):
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=units, activation=tf.nn.tanh, name="RNN_cell")
    _, last_states = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, sequence_length=sequence_length, dtype=tf.float32)
    return last_states


def save_graph_summary(summary_file):
    summary_writer = tf.summary.FileWriter(summary_file)
    summary_writer.add_graph(tf.get_default_graph())
    summary_writer.flush()


def save_graph(graph, session, checkpoint):
    utils.io.ensure_parent_exists(checkpoint)
    with graph.as_default():
        saver = tf.train.Saver()
        saver.save(session, checkpoint)


def load_graph(graph, session, checkpoint):
    with graph.as_default():
        saver = tf.train.Saver()
        saver.restore(session, checkpoint)


def get_train_op(loss, learning_rate=0.001):
    with tf.name_scope('Train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # This has to be done AFTER adding all BN layers to graph
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients = [None if gradient is None else tf.clip_by_norm(gradient, 3.0)
                         for gradient in gradients]
            train_op = optimizer.apply_gradients(zip(gradients, variables))
            # train_op = optimizer.minimize(loss)
        _ = tf.global_variables_initializer()
    return train_op
