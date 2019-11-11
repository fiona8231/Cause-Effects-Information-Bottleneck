#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import edward as ed
import tensorflow as tf
from edward.models import Bernoulli, Normal
from datasets import IHDP
from evaluation import Evaluator
import numpy as np
import time
from scipy.stats import sem

from utils import fullyConnect_net, get_y0_y1
from argparse import ArgumentParser

best_logpvalid = - np.inf

parser = ArgumentParser()
parser.add_argument('-reps', type=int, default=10)
parser.add_argument('-earl', type=int, default=10)
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-opt', choices=['adam', 'adamax'], default='adam')
parser.add_argument('-epochs', type=int, default=100)
parser.add_argument('-print_every', type=int, default=10)
args = parser.parse_args()

args.true_post = True

ed.set_seed(1)
np.random.seed(1)
tf.set_random_seed(1)

dataset = IHDP(replications=args.reps)
scores = np.zeros((args.reps, 3))
scores_test = np.zeros((args.reps, 3))

M = None
d = 20  # latent space dimension
lamba = 1e-4  # weight decay
nh, h = 5, 200  # number and size of hidden layers

for i, (train, valid, test, contfeats, binfeats) in enumerate(dataset.get_train_valid_test()):
    print('\nReplication {}/{}'.format(i + 1, args.reps))
    (xtr, ttr, ytr), (y_cftr, mu0tr, mu1tr) = train
    (xva, tva, yva), (y_cfva, mu0va, mu1va) = valid
    (xte, tte, yte), (y_cfte, mu0te, mu1te) = test
    evaluator_test = Evaluator(yte, tte, y_cf=y_cfte, mu0=mu0te, mu1=mu1te)

    # Reorder features with binary first and continuous after
    perm = binfeats + contfeats
    xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]

    xalltr, talltr, yalltr = np.concatenate([xtr, xva], axis=0), np.concatenate([ttr, tva], axis=0), np.concatenate([ytr, yva], axis=0)

    evaluator_train = Evaluator(yalltr, talltr, y_cf=np.concatenate([y_cftr, y_cfva], axis=0),
                                mu0=np.concatenate([mu0tr, mu0va], axis=0), mu1=np.concatenate([mu1tr, mu1va], axis=0))

    # zero mean, unit variance for y during training
    ym, ys = np.mean(ytr), np.std(ytr)
    ytr, yva = (ytr - ym) / ys, (yva - ym) / ys
    best_logpvalid = - np.inf

    with tf.Graph().as_default():
        sess = tf.InteractiveSession()

        ed.set_seed(1)
        np.random.seed(1)
        tf.set_random_seed(1)

        x_ph_bin = tf.placeholder(tf.float32, [M, len(binfeats)], name='x_bin')  # binary inputs
        x_ph_cont = tf.placeholder(tf.float32, [M, len(contfeats)], name='x_cont')  # continuous inputs
        t_ph = tf.placeholder(tf.float32, [M, 1])
        y_ph = tf.placeholder(tf.float32, [M, 1])

        x_ph = tf.concat([x_ph_bin, x_ph_cont], 1)

        activation = tf.nn.elu

        # p(z) -> Define prior  N(0,I)
        z = Normal(loc=tf.zeros([tf.shape(x_ph)[0], d]), scale=tf.ones([tf.shape(x_ph)[0], d]))

        # *********************** Decoder start from here ***********************

        # p(t|z)   nh, h = 5, 200
        logits = fullyConnect_net(z, [h], [[1, None]], 'pt_z', lamba=lamba, activation=activation)
        t = Bernoulli(logits=logits, dtype=tf.float32)

        # p(y|t,z)
        mu2_t0 = fullyConnect_net(z, nh * [h], [[1, None]], 'py_t0z', lamba=lamba, activation=activation)
        mu2_t1 = fullyConnect_net(z, nh * [h], [[1, None]], 'py_t1z', lamba=lamba, activation=activation)
        y = Normal(loc=t * mu2_t1 + (1. - t) * mu2_t0, scale=tf.ones_like(mu2_t0))

        # *********************** Encoder start from here ***********************

        muq_t0, sigmaq_t0 = fullyConnect_net(x_ph, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt0', lamba=lamba,
                                             activation=activation)

        qz = Normal(loc=muq_t0, scale=sigmaq_t0)

        # Sampling posterior predictive from p(y|z,t)
        y_post = ed.copy(y, {z: qz, t: t_ph}, scope='y_post')
        t_post = ed.copy(t, {z: qz, y: y_ph}, scope='t_post')

        # for early stopping according to a validation set
        y_post_eval = ed.copy(y, {z: qz.mean(), y: y_ph, t: t_ph}, scope='y_post_eval')

        t_post_eval = ed.copy(t, {z: qz.mean(), y: y_ph}, scope='t_post_eval')

        log_valid = tf.reduce_mean(tf.reduce_sum(y_post_eval.log_prob(y_ph) + t_post_eval.log_prob(t_ph), axis=1) + tf.reduce_sum(z.log_prob(qz.mean()) - qz.log_prob(qz.mean()), axis=1) )

        tf.global_variables_initializer().run()

        # Information bottleneck control parameter
        BETA =16671.79 #257.83 #2753.05 #9268.75 #4806.3 #16671.79

        # Latent Loss
        info_loss = tf.reduce_sum(tf.contrib.distributions.kl_divergence(qz, z))

        # Log-Likelihood
        class_loss = - BETA * tf.reduce_sum(y_post.log_prob(y_ph) + t_post.log_prob(t_ph), axis=1)

        # Define Loss Funciton
        total_loss = tf.reduce_mean(class_loss + info_loss)

        # Training optimizer
        train_op = tf.train.AdamOptimizer(1e-4).minimize(total_loss)

        saver = tf.train.Saver(tf.contrib.slim.get_variables())
        tf.global_variables_initializer().run()

        n_epoch, n_iter_per_epoch, idx = args.epochs, 10 * int(xtr.shape[0] / 100), np.arange(xtr.shape[0])

        # Dictionaries needed for evaluation
        tr0, tr1 = np.zeros((xalltr.shape[0], 1)), np.ones((xalltr.shape[0], 1))
        tr0t, tr1t = np.zeros((xte.shape[0], 1)), np.ones((xte.shape[0], 1))
        f1 = {x_ph_bin: xalltr[:, 0:len(binfeats)], x_ph_cont: xalltr[:, len(binfeats):], t_ph: tr1}
        f0 = {x_ph_bin: xalltr[:, 0:len(binfeats)], x_ph_cont: xalltr[:, len(binfeats):], t_ph: tr0}
        f1t = {x_ph_bin: xte[:, 0:len(binfeats)], x_ph_cont: xte[:, len(binfeats):], t_ph: tr1t}
        f0t = {x_ph_bin: xte[:, 0:len(binfeats)], x_ph_cont: xte[:, len(binfeats):], t_ph: tr0t}

        for epoch in range(n_epoch):

            t0 = time.time()

            np.random.shuffle(idx)

            # ************** Start Training *************
            for j in range(n_iter_per_epoch):

                batch = np.random.choice(idx, 100)
                x_train, y_train, t_train = xtr[batch], ytr[batch], ttr[batch]

                sess.run((train_op, total_loss, info_loss, class_loss, log_valid, qz),
                         feed_dict={x_ph_bin: x_train[:, 0:len(binfeats)], x_ph_cont: x_train[:, len(binfeats):], t_ph: t_train, y_ph: y_train})

            if epoch % args.earl == 0 or epoch == (n_epoch - 1):
                logpvalid = sess.run(log_valid, feed_dict={x_ph_bin: xva[:, 0:len(binfeats)], x_ph_cont: xva[:, len(binfeats):],
                                                           t_ph: tva, y_ph: yva})
                # Early stopping prevent overfitting
                if logpvalid >= best_logpvalid:
                    print('Improved Validation Bound, Old: {:0.3f}, New: {:0.3f}'.format(best_logpvalid, logpvalid))
                    best_logpvalid = logpvalid
                    # saving model
                    saver.save(sess, 'models/ihdp')

            if epoch % args.print_every == 0:
                y0, y1 = get_y0_y1(sess, y_post, f0, f1, shape=yalltr.shape, L=1)
                y0, y1 = y0 * ys + ym, y1 * ys + ym
                score_train = evaluator_train.calc_stats(y1, y0)
                rmses_train = evaluator_train.y_errors(y0, y1)

                y0, y1 = get_y0_y1(sess, y_post, f0t, f1t, shape=yte.shape, L=1)
                y0, y1 = y0 * ys + ym, y1 * ys + ym
                score_test = evaluator_test.calc_stats(y1, y0)

                print("Epoch: {}/{}, Validation Bound >= {:0.3f}, ICE_train: {:0.3f}, ACE_train: {:0.3f}, " 
                      "ICE_test: {:0.3f}, ACE_test: {:0.3f}, BETA = {:0.2f}" .format(epoch + 1, n_epoch, logpvalid,
                                                  score_train[0], score_train[1], score_test[0], score_test[1], BETA))

        saver.restore(sess, 'models/ihdp')
        y0, y1 = get_y0_y1(sess, y_post, f0, f1, shape=yalltr.shape, L=100)
        y0, y1 = y0 * ys + ym, y1 * ys + ym
        score = evaluator_train.calc_stats(y1, y0)
        scores[i, :] = score

        y0t, y1t = get_y0_y1(sess, y_post, f0t, f1t, shape=yte.shape, L=100)
        y0t, y1t = y0t * ys + ym, y1t * ys + ym
        score_test = evaluator_test.calc_stats(y1t, y0t)
        scores_test[i, :] = score_test

        print('Replication: {}/{}, Train_ICE: {:0.3f}, Train_ACE: {:0.3f},' 
              'Test_ICE: {:0.3f}, Test_ACE: {:0.3f} '.format(i + 1, args.reps, score[0], score[1], score_test[0], score_test[1],))
        sess.close()

    print('************ Finish ************')
    print('CEIB Model Total Scores:')
    means, stds = np.mean(scores, axis=0), sem(scores, axis=0)
    print(' Train ACE: {:.3f}+-{:.3f}' ''.format(means[1], stds[1]))

    means, stds = np.mean(scores_test, axis=0), sem(scores_test, axis=0)
    print(' Test  ACE: {:.3f}+-{:.3f}' ''.format(means[1], stds[1]))
