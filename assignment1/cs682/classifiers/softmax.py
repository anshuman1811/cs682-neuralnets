import numpy as np
from random import shuffle

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
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]

  for i in range(N):
    scores = np.zeros(C)
    dadw = np.zeros_like(W) # CxN - a=np.exp(scores[y[i]])
    dbdw = np.zeros_like(W) # CxN - b=np.sum(np.exp(scores))
    for j in range(C):
      for k in range(D):
        scores[j] += X[i][k]*W[k][j] 
      dbdw[:,j] += X[i]*np.exp(scores[j]) # 1xD
      if j == y[i]:
        dadw[:,j] += X[i]*np.exp(scores[j]) # 1xD
    
    a = np.exp(scores[y[i]])
    b = np.sum(np.exp(scores))
    k = a/b
    dkda = 1/b
    dkdb = -a/(b*b)
    dkdw = ((dadw)*b - (dbdw)*a)/(b*b)
    
    dLidk = -1/k
    dLidw = dLidk*dkdw
    loss += -1.0*np.log(k)
    dW += dLidw
  
  loss/=N
  dW/=N

  loss += reg*(np.sum(np.square(W)))
  dW += 2*reg*W
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
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]

  scores = X.dot(W) # NxC
  scores -= np.max(scores, axis=1, keepdims=True)
  scores_correct_class = scores[np.arange(N), y] 
  
  # adding logC = -(max score) for numeric stability
  a = np.exp(scores_correct_class)
#   ai = np.exp(scores_correct_class[i])
  b = np.sum(np.exp(scores), axis = 1)
#   bi = np.exp(np.exp(scores[i]), axis = 1)

  k = a/b
  loss = -np.sum(np.log(k))
  helper_matrix = np.exp(scores)/np.reshape(b, (-1,1))
  helper_matrix[np.arange(N), y] -= 1
  dW = X.T.dot(helper_matrix)

  loss /= N
  dW /= N
    
  loss += reg*np.sum(np.square(W))
  dW += 2*reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

