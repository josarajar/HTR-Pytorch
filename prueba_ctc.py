from torch import nn
import torch
import tensorflow as tf
import math
import numpy as np

time_step = 50  # Input sequence length
vocab_size = 20  # Number of classes
batch_size = 16  # Batch size
target_sequence_length = 30  # Target sequence length


def dense_to_sparse(dense_tensor, sequence_length):
    indices = tf.where(tf.sequence_mask(sequence_length))
    values = tf.gather_nd(dense_tensor, indices)
    shape = tf.shape(dense_tensor, out_type=tf.int64)
    return tf.SparseTensor(indices, values, shape)


def target_list_to_sparse_tensor(targetList):
    """
    Make tensorflow SparseTensor from list of targets, with each element in
    the list being a list or array with the values of the target sequence

    Args:
        targetList: list of target sequences

    Returns:
        tuple containing three elements: array of indices, array of vals and
            array containing the shape

    """
    import numpy as np
    indices = []
    vals = []
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            vals.append(val)
    shape = [len(targetList), np.asarray(indices).max(0)[1] + 1]
    return (np.array(indices), np.array(vals), np.array(shape))

def compute_loss(y, x, x_len):
    ctclosses = tf.nn.ctc_loss(
        y,
        tf.cast(x, dtype=tf.float32),
        x_len,
        preprocess_collapse_repeated=False,
        ctc_merge_repeated=True,
    )
    ctclosses = tf.reduce_mean(ctclosses)

    with tf.Session() as sess:
        ctclosses = sess.run(ctclosses)
        print(f"tf ctc loss: {ctclosses}")


minimum_target_length = 10

# Para que el resultado de CTC en pytorch salga igual que en Tensorflow, hay que llamar a ctc con el parámetro reduction='None', que devolverá el coste
# por cada secuencia, y luego aplicar la media al igual que se hacía en tensorflow. Es posible en pytorch, si se configura dicho
# parámetro con reduction='mean', que calcule la media pero ponderando por la longitud de las secuencias target (ground truth) y luego obtiene la media.
ctc_loss = nn.CTCLoss(blank=vocab_size - 1, reduction='none')
x = torch.randn(time_step, batch_size, vocab_size)  # [size] = T,N,C
#y = torch.randint(0, vocab_size - 2, (batch_size, target_sequence_length), dtype=torch.long)  # low, high, [size]

x_lengths = torch.full((batch_size,), time_step, dtype=torch.long)  # Length of inputs
y_lengths = torch.randint(minimum_target_length, target_sequence_length, (batch_size,),
                          dtype=torch.long)  # Length of targets can be variable (even if target sequences are constant length)

y = torch.randint(0, vocab_size - 2, (torch.sum(y_lengths),), dtype=torch.long)  # low, high, [size]

loss = torch.mean(ctc_loss(x.log_softmax(2).detach(), y, x_lengths, y_lengths))
print(f"torch ctc loss: {loss}")

x = x.log_softmax(2).detach().numpy()
y = y.numpy()
x_lengths = x_lengths.numpy()
y_lengths = y_lengths.numpy()
x = tf.cast(x, dtype=tf.float32)
#y = tf.cast(dense_to_sparse(y, y_lengths), dtype=tf.int32)
y = tf.cast(tf.SparseTensor(*target_list_to_sparse_tensor([y[np.sum(y_lengths[:ind]):np.sum(y_lengths[:ind+1])] for ind in range(y_lengths.shape[0])])), dtype=tf.int32)
compute_loss(y, x, x_lengths)