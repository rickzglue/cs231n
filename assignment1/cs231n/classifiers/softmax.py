import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in xrange(num_train):
    scores = X[i].dot(W)

    # Numerical stability - shift the score values
    #
    scores -= np.max(scores)

    sum_exp = np.sum( np.exp(scores) )  # needed for gradient
    loss += -scores[y[i]] + np.log( sum_exp )

    # gradient = derivative of loss function w.r.t 1 classifier
    # 
    # loss = -f_yi + log( sum( exp(f_j) ) )
    #
    # gradient_j!=y -> 1/sum_exp * exp(f_j)
    # gradient_y    -> -1 + 1/sum_exp * exp(f_y)
    #
    for j in xrange(num_classes):
        dW[:,j] += 1.0/sum_exp * np.exp( scores[j] ) * X[i]
        if j == y[i]:
            dW[:,j] -= X[i]
  
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # average gradients
  dW  /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # Add regularization to the gradiant
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  xInd      = np.arange(num_train)
  
  scores = X.dot(W)

  # Numerical stability - shift the score values
  #
  scores -= np.max(scores)
  sum_exp = np.sum( np.exp(scores), axis=1 )  # needed for gradient
  loss_per_ex = -scores[xInd, y] + np.log( sum_exp )
  loss = np.sum( loss_per_ex, axis=0 ) # sum across the examples

  # For Gradient, score * 1/sum_exp
  #
  exp_scores = np.exp(scores)
  norm_exp_scores = 1.0/sum_exp * exp_scores.T # C x N for batch
  norm_exp_scores = norm_exp_scores.T # N x C

  # For correct class, -1
  #
  norm_exp_scores[xInd, y] -= 1
 
  # dW = X.T * norm_exp_scores
  #
  dW = (X.T).dot(norm_exp_scores)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # average gradients
  dW  /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # Add regularization to the gradiant
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

