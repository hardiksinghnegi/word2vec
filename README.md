# word2vec 
This repository contains the implementation of word2vec algorithm to produce word embeddings using neural models.

## Implementation Details:

1. word2vec_basic.py

-> In this file generate_batch(data, batch_size, num_skips, skip_window) function was implemented to return list of
context words (batch) and predicted words (labels) of length = batch_size.
-> I extracted the elements of window size from the array by calculating window_size = 2*skip_window+1. Then we define the
position of the context word context_index = len(win_grab)//2.
-> I extracted word ids from left and right of the context word and the no of extracted word ids is equal to num_skips.
-> After, this the pair of (context word, predicted word) was inserted in the batch_list and then these values are added to
the np arrays batch and labels and returned for further processing.

2. loss_func.py

In this file 2 loss functions were implemented as follows:

a) Cross Entropy Loss

-> For this I implemented cross_entropy_loss(inputs, true_w) function.
-> Then calculate two values A, which simply calculates log of exponent of dot product of u_o (context word) and v_c
(predicted word). Here the use tf.einsum() is to calculate dot product between u_o and v_c, tf.exp() to calculate exponent
and tf.log to calculate final logarithmic value.
-> For second value B calculate log of summation of exponent of matrix multiplication of u_w (inputs) and v_c (true_w).
-> Final loss value is retured as difference of B and A calulated as tf.subtract(B,A)

b) NCE Loss

-> For this nce_loss(inputs, weights, biases, labels, sample, unigram_prob) function is implemented
-> The formula for NCE is −[logPr(D = 1,wo|wc)+ 􏰀Summation(log(1−Pr(D = 1,wx|wc)))] I divided this in two parts, in which
part A involves calculation of logPr(D = 1,wo|wc) and part B involves calculation of 􏰀Summation(log(1−Pr(D = 1,wx|wc)))
-> For calculating part A I followed almost similar approach taken in calculating A in cross_entropy_loss function i.e.
use of tf.einsum() for dot product of u_c (inputs) and u_o (labels) and then addition biases b_o too.
->Here I had to convert few np arrays like unigram_prob and sample to tensor using tf.convert_to_tensor() for further
processing. I also performed lookup for vector embeddings using tf.nn.embedding_lookup() and give them proper shape using
tf.reshape(), wherever needed.
->For calculating part B Summation(log(1−Pr(D = 1,wx|wc))) was derieved. This instead of dot product uses tf.matmul() for
complete matrix multiplication. For calculating sigmoid function tf.sigmoid() was used while working on both part A and B.
-> In this calculation of NCE loss, to be compensate for nan values I added small constant tensors of order 10^-8 to
other tensors before taking their log.
-> In the end -1*sum(partA,partB) is returned as NCE loss.

3. word_analogy.py

-> In this file I calculated cosine spatial distance to derive least and most illustrative pair among the list of words
provided.
-> I extracted lines from word_analogy_dev.txt and parse them for the pair of words on either side of '||'. Then
calculated average of the vector difference of word embeddings on the left using np.mean(). After this calculate 1 - cosine
spatial distance of this average with vector difference of every word pair embedding on right.
-> To get spatial distance I used : 1 - spatial.distance.cosine(<vect1>, <vect2>). The pair with min distance is the least
illustrative pair and the pair with max spatial distance is most illustrative pair.

4. Calculating top 20 words (top20.py)

-> For calculating top 20 words first loopkup for embeddings of words a) first b) american c) would was done.
-> Then I created a dictionary embedd_word{} that has (word, embedding) key pair for every word in dictionary. Then calculate
spatial distance of each one of first, american and would. Expression used: cosine_dist = 1 - spatial.distance.cosine(vect1, vect2)
-> Then store this in another dictionary in form of ( cosine_dist, word) key value pair. Then this dictionary is sorted
according to descending order values of cosine_dist. Choose top 20 words in this dictionary excluding the comparision
word itself. For this I used collections.OrderedDict() for sorted(dict.items(), reverse=True).
-> Calculated top 20 words for Cross Entropy Loss as well as NCE Loss model.

## Requirements
- Python2.7
- TensorFlow: 1.11.0
- Linux / Mac OS HighSierra

## Commands To Run
To generate model files run the following commands:

1) For Cross Entropy Model:

python word2vec_basic.py

This will generate word2vec_cross_entropy.model file

The best model generated is for the tweaking of hyperparameters: skip_window = 8 and num_skips = 16 and learning rate = 0.3

2) For NCE model:

python word2vec_basic.py nce

This will generate word2vec_nce.model file

The best model generated is for the tweaking of hyperparameters: batch_size = 256 and learning rate = 0.3

3) For word_analogy.py:

Command is : python word_analogy.py

For generating predictions for Cross entropy set variable loss_model = 'cross_entropy' and
output file = 'word_analogy_test_predictions_cross_entropy.txt'

For generating predictions for NCE set variable loss_model = 'nce' and
output file = 'word_analogy_test_predictions_nce.txt'

The result of accuracy can be determined by using command:
./score_maxdiff.pl word_analogy_dev_mturk_answers.txt {output prediction file} {result file}

4) For getting top 20 words just run python top20.py
and set the loss_model = 'cross_entropy' or 'nce'
