#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import edward as ed
import tensorflow as tf
from edward.models import Bernoulli, Normal
from datasets import IHDP
from evaluation import Evaluator
import numpy as np


from utils import fullyConnect_net, get_y0_y1
from argparse import ArgumentParser
from util.IOTools import IOTools
from Plot import plot_information_curve_line, plot_information_curve_scatter

import matplotlib
matplotlib.use('Agg')
best_logpvalid = - np.inf


parser = ArgumentParser()
parser.add_argument('-reps', type=int, default=1)
parser.add_argument('-earl', type=int, default=1)
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-opt', choices=['adam', 'adamax'], default='adam')
parser.add_argument('-epochs', type=int, default=100)
parser.add_argument('-print_every', type=int, default=3)
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

    # reorder features with binary first and continuous after
    perm = binfeats + contfeats
    xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]

    xalltr, talltr, yalltr = np.concatenate([xtr, xva], axis=0), np.concatenate([ttr, tva], axis=0), np.concatenate([ytr, yva], axis=0)
    evaluator_train = Evaluator(yalltr, talltr, y_cf=np.concatenate([y_cftr, y_cfva], axis=0),
                                mu0=np.concatenate([mu0tr, mu0va], axis=0), mu1=np.concatenate([mu1tr, mu1va], axis=0))

    # zero mean, unit variance for y during training
    ym, ys = np.mean(ytr), np.std(ytr)
    ytr, yva = (ytr - ym) / ys, (yva - ym) / ys
    best_logpvalid = - np.inf

    path = '../IBCE-beta/' + 'pickle.pkl'
    with tf.Graph().as_default():
        sess = tf.InteractiveSession()

        ed.set_seed(1)
        np.random.seed(1)
        tf.set_random_seed(1)

        n_epoch, n_iter_per_epoch, idx = args.epochs, 10 * int(xtr.shape[0] / 100), np.arange(xtr.shape[0])
        batch = 100

        # M = None -> batch size during training
        x_ph_bin = tf.placeholder(tf.float32, [M, len(binfeats)], name='x_bin')  # binary inputs
        x_ph_cont = tf.placeholder(tf.float32, [M, len(contfeats)], name='x_cont')  # continuous inputs
        t_ph = tf.placeholder(tf.float32, [M, 1])
        y_ph = tf.placeholder(tf.float32, [M, 1])

        beta_holder = tf.placeholder('float32', [1, 1])
        x_ph = tf.concat([x_ph_bin, x_ph_cont], 1)
        activation = tf.nn.elu

        # p(z) -> define prior
        z = Normal(loc=tf.zeros([tf.shape(x_ph)[0], d]), scale=tf.ones([tf.shape(x_ph)[0], d]))

        # *********************** Decoder start from here ***********************

        # p(t|z)   nh, h = 3, 200
        logits = fullyConnect_net(z, [h], [[1, None]], 'pt_z', lamba=lamba, activation=activation)
        t = Bernoulli(logits=logits, dtype=tf.float32)

        # p(y|t,z)
        mu2_t0 = fullyConnect_net(z, nh * [h], [[1, None]], 'py_t0z', lamba=lamba, activation=activation)
        mu2_t1 = fullyConnect_net(z, nh * [h], [[1, None]], 'py_t1z', lamba=lamba, activation=activation)
        y = Normal(loc=t * mu2_t1 + (1. - t) * mu2_t0, scale=tf.ones_like(mu2_t0))

        # *********************** Encoder start from here ***********************

        muq_t0, sigmaq_t0 = fullyConnect_net(x_ph, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt0', lamba=lamba, activation=activation)

        # Latent space
        qz = Normal(loc=muq_t0, scale=sigmaq_t0)

        # sample posterior predictive for p(y|z,t)
        y_post = ed.copy(y, {z: qz, t: t_ph}, scope='y_post')
        t_post = ed.copy(t, {z: qz, y: y_ph}, scope='t_post')

        y_post_eval = ed.copy(y, {z: qz.mean(), y: y_ph, t: t_ph}, scope='y_post_eval')

        t_post_eval = ed.copy(t, {z: qz.mean(), y: y_ph}, scope='t_post_eval')

        # Latent Loss
        info_loss = tf.reduce_sum(tf.contrib.distributions.kl_divergence(qz, z))
        # Initial information bottleneck parameter

        beta_value = 1

        # Likelihood
        class_loss = - beta_holder * tf.reduce_sum(y_post.log_prob(y_ph) + t_post.log_prob(t_ph), axis=1)

        # Define Loss Funciton
        total_loss = tf.reduce_mean(class_loss + info_loss)

        # Training optimizer
        train_op = tf.train.AdamOptimizer(1e-4).minimize(total_loss)

        saver = tf.train.Saver(tf.contrib.slim.get_variables())
        tf.global_variables_initializer().run()

        # Dictionaries needed for evaluation
        tr0, tr1 = np.zeros((xalltr.shape[0], 1)), np.ones((xalltr.shape[0], 1))
        tr0t, tr1t = np.zeros((xte.shape[0], 1)), np.ones((xte.shape[0], 1))
        f1 = {x_ph_bin: xalltr[:, 0:len(binfeats)], x_ph_cont: xalltr[:, len(binfeats):], t_ph: tr1}
        f0 = {x_ph_bin: xalltr[:, 0:len(binfeats)], x_ph_cont: xalltr[:, len(binfeats):], t_ph: tr0}
        f1t = {x_ph_bin: xte[:, 0:len(binfeats)], x_ph_cont: xte[:, len(binfeats):], t_ph: tr1t}
        f0t = {x_ph_bin: xte[:, 0:len(binfeats)], x_ph_cont: xte[:, len(binfeats):], t_ph: tr0t}

        ixz = list()
        izty = list()
        h_y = list()
        beta_list = list()

        iteration = 250000

        # ************** Start Training *************
        for epoch in range(iteration):

            batch = np.random.choice(idx, 100)
            x_train, y_train, t_train = xtr[batch], ytr[batch], ttr[batch]

            # sample new batch
            _, totalloss, infoloss, reconloss = sess.run((train_op, total_loss, info_loss, class_loss),
                                                         feed_dict={x_ph_bin: x_train[:, 0:len(binfeats)],
                                                             x_ph_cont: x_train[:, len(binfeats):], t_ph: t_train,
                                                             y_ph: y_train, beta_holder: np.asarray([[beta_value]])})

            if epoch % 200 == 0 and epoch > 0:  # 50

                print("Iteration: %d, KL div: %0.4f" % (epoch, np.mean(infoloss)))

                # Set the lower bound for I(X,Z)
                if np.mean(infoloss) > 0.1 and len(h_y) > 0:  # Only calc this part if already got some entropy,

                    # save MI(x,z)
                    ixz.append(np.mean(infoloss))
                    beta_list.append(beta_value)

                    entropy = np.mean(np.absolute(np.asarray(h_y)))
                    print("Cost: %.2f, I(X;Z): %.4f, I(Z;(T,Y)): %.4f, BETA = %.2f" % (totalloss, np.mean(infoloss),
                                                                                      np.absolute(entropy - np.mean(reconloss) / beta_value),
                                                                                      beta_value))

                    # save MI(Z;(T,Y))
                    izty.append(np.absolute(entropy - np.mean(reconloss) / beta_value))

                    mi_x_t = np.asarray(ixz)
                    mi_t_y = np.asarray(izty)

                    nbins = int(min(12, max(1, np.floor(len(mi_x_t) / 3))))
                    breaks = np.linspace(0.99 * min(mi_x_t), max(mi_x_t), nbins + 1)

                    xl = list()
                    yl = list()
                    yl_means = list()

                    # For plotting the information curve line of mean
                    for k in range(nbins):
                        matchings_indices = [i for i, item in enumerate(mi_x_t) if
                                             item > breaks[k] and item < breaks[k + 1]]

                        # if more than 3 MI -> create new bin
                        if len(matchings_indices) > 3:
                            xl.append(np.mean(mi_x_t[matchings_indices]))
                            yl.append(mi_t_y[matchings_indices])
                            yl_means.append(np.median(mi_t_y[matchings_indices]))

                else:
                    if np.mean(infoloss) < 0.1:  # save the entropy if I(X;Z) is 0.

                        # collect mutual information in order to calculate the empirical entropy of Y

                        print("Collect Entropy... KL Div: %0.4f" % np.mean(infoloss))

                        h_y.append((np.mean(reconloss) / beta_value))

                # Increase Beta
                beta_value = beta_value * 1.01

    IOTools.save_to_file((yl_means, yl, xl, izty, ixz, batch, beta_list), path)
    # Plot both scatter and info curve line
    plot_information_curve_line('pickle.pkl', 'models/', ixz, izty)

    # Only plot scatter
    plot_information_curve_scatter(ixz, izty)

    sess.close()

    print('Finished!')

    print('Maximum Lambda Value: {:.3f}'
          ''.format(beta_value))
