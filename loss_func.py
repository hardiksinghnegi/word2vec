import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    '''Calculate dot product of context word u_o and target word v_c'''
    mul_A = tf.einsum('ij, ij->i', inputs, true_w)
    '''Calulation of A'''
    A = tf.log(tf.exp(mul_A))

    '''Calculation of B'''
    mul_B = tf.matmul(inputs, tf.transpose(true_w))
    B = tf.log(tf.reduce_sum(tf.exp(mul_B), axis=1))

    '''Return Cross Entropy Loss'''
    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
    '''Convert sample and unigram_prob to tensor for further processing'''
    sample = tf.convert_to_tensor(sample, dtype=tf.int32)
    unigram_prob = tf.convert_to_tensor(unigram_prob, dtype=tf.float32)

    '''Extract batch_size, embedding_size and sample size(k)'''
    batch_size, embedding_size = inputs.get_shape().as_list()
    k = int(sample.get_shape()[0])

    '''Preparing tensors calculating logPr(D = 1,wo|wc)'''
    '''u_c is the context words'''
    u_c = inputs
    '''Lookup embedding for target word u_o'''
    u_o = tf.reshape(tf.nn.embedding_lookup(weights, labels), shape=[batch_size, embedding_size])
    '''Lookup bias'''
    b_o = tf.nn.embedding_lookup(biases, labels)

    '''Caluculate Dot product of context word u_c and target word u_o and add bias to it'''
    dot_prod = tf.einsum('ij, ij->i', u_c, u_o)
    part_a = tf.add(tf.reshape(dot_prod, shape=[batch_size, 1]), b_o)

    '''Calculate log [kPr(wo)]) by looking up unigram probability pr_wo'''
    pr_wo = tf.nn.embedding_lookup(unigram_prob, labels)
    part_b = tf.log(tf.scalar_mul(k,pr_wo))

    '''Prepare tensors by embedding lookup of weights'''
    u_x = tf.reshape(tf.nn.embedding_lookup(weights, sample), shape=[k, embedding_size])
    bt_x = tf.transpose(tf.reshape(tf.nn.embedding_lookup(biases, sample), shape=[k, 1]))
    bt_x = tf.ones([batch_size, 1])*bt_x

    '''Calculate negative sample'''
    part_ax = tf.add(tf.matmul(u_c, tf.transpose(u_x)), bt_x)

    pr_wx = tf.transpose(tf.reshape(tf.nn.embedding_lookup(unigram_prob, sample), shape=[k, 1]))
    prt_wx = tf.ones([batch_size, 1])*pr_wx

    '''Calculating smoothing tensors to prevent nan values'''
    smoothing_tensor_1 = tf.scalar_mul(0.00000001, tf.ones([batch_size, 1]))
    smoothing_tensor = tf.scalar_mul(0.00000001, tf.ones([batch_size, k]))

    part_bx = tf.log(tf.scalar_mul(k, prt_wx))
    sigma_x = tf.sigmoid(tf.subtract(part_ax, part_bx))

    x_1 = tf.subtract(part_a, part_b)

    cost_A = tf.log(tf.add((tf.sigmoid(x_1)), smoothing_tensor_1))

    one_t = tf.ones([batch_size, k])
    sigma_x = tf.add(tf.subtract(one_t, sigma_x), smoothing_tensor)
    cost_B = tf.reshape(tf.reduce_sum(tf.log(sigma_x), axis=1), [batch_size, 1])

    '''Return NCE loss value'''
    return tf.scalar_mul(-1, tf.add(cost_A, cost_B))