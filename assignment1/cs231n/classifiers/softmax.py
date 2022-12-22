from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0] # X_dev.shape[0] = 500 -> 500개의 데이터
    num_classes = W.shape[1] # X_dev.shape[1] = 10 -> 10개의 클래스
    for i in range(num_train): # 500번 반복
        scores = X[i].dot(W) # (i, 3073) * (3073, 10) = (i, 10)
        scores -= np.max(scores) # overflow 방지
        correct_class_score = scores[y[i]] # 정답 클래스의 점수
        exp_sum = np.sum(np.exp(scores)) # 모든 클래스의 점수의 합
        loss += -correct_class_score + np.log(exp_sum) # loss 계산, 500개의 loss의 합
        for j in range(num_classes): # 10번 반복
            dW[:, j] += X[i] * np.exp(scores[j]) / exp_sum # gradient 계산
        dW[:, y[i]] -= X[i] # 정답 클래스의 gradient 계산

    # 평균 loss, gradient 계산
    loss /= num_train # 평균 loss 계산, loss = -log(softmax)
    # softmax의 loss는 -log(softmax)인 이유, https://www.youtube.com/watch?v=OoUX-nOEjG0

    # 아래 loss는 작동하지 않음
    loss += reg * np.sum(W * W) # regularization loss 계산, loss = -log(softmax) + reg * W^2
    # loss가 -log(0.1) = 2.3 정도로 나오는데, 이는 정답 클래스의 점수가 0.1이라는 뜻이다.
    # 이는 softmax 함수의 결과가 0.1이라는 뜻이다.

    dW /= num_train # 평균 gradient 계산
    dW += 2 * reg * W # regularization gradient 계산
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0] # X_dev.shape[0] = 500 -> 500개의 데이터
    scores = X.dot(W) # (500, 3073) * (3073, 10) = (500, 10)
    scores -= np.max(scores, axis=1, keepdims=True) # overflow 방지
    correct_class_score = scores[np.arange(num_train), y] # 정답 클래스의 점수
    exp_sum = np.sum(np.exp(scores), axis=1) # 모든 클래스의 점수의 합
    loss = np.sum(-correct_class_score + np.log(exp_sum)) # loss 계산
    loss /= num_train # 평균 loss 계산
    loss += reg * np.sum(W * W) # regularization loss 계산
    softmax = np.exp(scores) / exp_sum.reshape(-1, 1) # softmax 계산
    softmax[np.arange(num_train), y] -= 1 # 정답 클래스의 gradient 계산
    dW = X.T.dot(softmax) # gradient 계산
    dW /= num_train # 평균 gradient 계산
    dW += 2 * reg * W # regularization gradient 계산
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
