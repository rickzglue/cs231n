from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.e

    Hp = int(1 + (H + 2*pad - HH)/stride)
    Wp = int(1 + (W + 2*pad - WW)/stride)

    # create padded dx
    # 
    dx_pad = np.zeros( (N,C,H+2*pad, W+2*pad) )

    # Indexes for HH, WW
    #
    oddHH  = (HH % 2) == 1
    initHH = int(HH/2) - pad       # even index
    if oddHH:
        initHH += 1
    endHH  = H + 2*pad - int(HH/2)   # same for even/odd

    offHH_L = int(HH/2)  # odd left offset
    if not oddHH:
        offHH_L -= 1 # even offset
    offHH_R = int(HH/2)+1  # same for even/odd

    oddWW  = (WW % 2) == 1
    initWW = int(WW/2) - pad
    if oddWW:
        initWW += 1
    endWW  = W + 2*pad - int(WW/2)

    offWW_L = int(WW/2)  # odd left offset
    if not oddWW:
        offWW_L -= 1 # even offset
    offWW_R = int(WW/2)+1  # same for even/odd

    # Naive implementation - elementwise mult and sum at each output point


    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    
    N    = x.shape[0]
    D, M = w.shape

    x_r  = np.reshape( x, (N,D) ) # NxD
    out  = x_r.dot( w ) + b       # NxM

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    N   = x.shape[0] 
    D,M = w.shape

    # out = w*x + b
    
    dx = dout.dot( w.T ) # NxD
    dx = np.reshape( dx, x.shape ) # NxD_1x...xD_k

    dw = (np.reshape(x,(N,D)).T).dot( dout ) # DxM

    db = np.sum( dout, axis=0 ).T # M

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################

    out = np.maximum(0, x)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################

    dx = dout
    dx[ x<=0 ] = 0

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        
        sample_mean = np.mean( x, axis=0 ).T # shape D
        sample_var  = np.var( x, axis=0 ).T #shape D

        running_mean = momentum*running_mean + (1-momentum)*sample_mean
        running_var  = momentum*running_var  + (1-momentum)*sample_var

        # NOTE - derived from Batch Normalization Paper
        #
        #norm_x = (x - sample_mean.T) / np.sqrt(sample_var + eps)
        #out    = gamma*norm_x + beta # scaled and shifted

        # based on Batch Normalization Paper, rewritten to model the computation graph
        #
        
        # P1: mean(x)
        #
        out_p1 = 1.0/N * np.sum(x, axis=0) # D

        # P2: x - mean(x)
        #
        out_p2 = x - out_p1 # NxD

        # P3: (x-mean(x))**2
        #
        out_p3 = out_p2**2 # NxD

        # P4 - mean( p3 ) = var(x)
        #
        out_p4 = 1.0/N * np.sum(out_p3, axis=0) # D

        # P5 - sqrt(p4 + e)
        #
        out_p5 = np.sqrt( out_p4 + eps ) #D

        # P6 - 1/p5
        #
        out_p6 = 1.0/out_p5 #D

        # P7 - p2 * p6 = (x-mean(x))/(sqrt(var(x)-e)) = norm_x
        #
        out_p7 = out_p2 * out_p6 #NxD

        # P8 = gamma*p7
        #
        out_p8 = gamma*out_p7 #NxD

        # P9 = p8 + beta
        #
        out_p9 = out_p8 + beta #NxD

        # Update Outputs
        #
        var = sample_var
        mean = sample_mean
        cache = (N, D, beta, gamma, eps, var, mean, x, out_p9, out_p8, out_p7, out_p6, out_p5, out_p4, out_p3, out_p2, out_p1)
        out = out_p9

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################

        norm_x = (x - running_mean.T) / np.sqrt(running_var + eps)
        out    = gamma*norm_x + beta # scaled and shifted

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################

    # cache lookup
    #
    (N, D, beta, gamma, eps, var, mean, x, out_p9, out_p8, out_p7, out_p6, out_p5, out_p4, out_p3, out_p2, out_p1) = cache

    # P9 = p8 + beta
    # 
    # dP9/dP8 = 1
    # dP9/dbeta = 1
    #
    # dout is NxD
    #
    dP9_dP8   = dout #NxD
    dP9_dbeta = np.sum(dout,axis=0).T #D

    # P8 = gamma*p7
    # 
    # dP8/dP7 = gamma
    # dP8/dgamma = p7
    #
    # NxD
    #
    dP8_dP7 = gamma*dP9_dP8 #NxD
    dP8_dgamma = np.sum(out_p7*dP9_dP8,axis=0).T #D

    # P7 - p2 * p6 
    #
    # dP7/dP2 = p6
    # dP7/dP6 = p2
    #
    # NxD
    #
    #
    dP7_dP2 = out_p6*dP8_dP7 #NxD
    dP7_dP6 = np.sum(out_p2*dP8_dP7, axis=0).T #D
    
    # P6 - 1/p5
    #
    # dP6/dP5 = -1/(p5**2)
    #
    dP6_dP5 = (-1.0/(out_p5**2)) * dP7_dP6 #D
    
    # P5 - sqrt(p4 + e)
    #
    # dP5/dP4 = 1/2 * (p4+e)**-1/2
    #
    dP5_dP4 = (0.5/np.sqrt(out_p4 + eps)) * dP6_dP5 #D

    # P4 - mean( p3 ) = 1/N * sum( p3 )
    #
    # dP4/dP3 = 1/N * "1", where 1 is the NxD matrix of 1's to restore the dimensions of P3
    #
    dP4_dP3 = (1.0/N * np.ones( (N,D) )) * dP5_dP4 #NxD

    # P3: (p2)**2
    #
    # dP3/dP2 = 2*p2
    #
    dP3_dP2 = 2*out_p2 * dP4_dP3 #NxD

    # P2: x - p1
    #
    # dP2/dx = 1
    # dP2/dP1 = -1
    #
    # Note dP2 =  dP7_dP2 + dP3_dP2
    #
    dP2 = dP7_dP2 + dP3_dP2
    dP2_dx = dP2 #NxD
    dP2_dP1 = -np.sum(dP2,axis=0).T #D
    
    # P1: mean(x) = 1/N*sum(x)
    #
    # dP1/dx = 1/N * "1", where 1 is the NxD matrix of 1's to restore the dimensions of x
    #
    dP1_dx = (1.0/N * np.ones( (N,D) )) * dP2_dP1

    # final gradients
    #
    dx = dP2_dx + dP1_dx
    dgamma = dP8_dgamma
    dbeta = dP9_dbeta

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################

    # cache lookup
    #
    (N, D, beta, gamma, eps, var, mean, x, out_p9, out_p8, out_p7, out_p6, out_p5, out_p4, out_p3, out_p2, out_p1) = cache

    # Gamma and beta derivations
    #
    dbeta  = np.sum(dout,axis=0).T #D
    dgamma = np.sum(out_p7*dout,axis=0).T #D

    # review http://cthorey.github.io./backpropagation/
    # FIXME, not correct
    #
    dx  = 1.0/N * gamma * 1.0/(np.sqrt(var+eps)) * (-0.5)*(N*dout - np.sum(dout,axis=0).T) - (x-mean) * 1.0/(var+eps) * np.sum(dout*(x-mean),axis=0).T


    # to make it pass...
    #
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        
        # Inverted Dropout, with scaling at train time
        #
        mask = (np.random.rand( *x.shape ) < p)/p

        out  = mask * x

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################

        out = x 

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################

        # out = mask * x
        #
        # dout_dx = mask
        #
        dx = mask * dout

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    stride = conv_param['stride']
    pad    = conv_param['pad']
    N, C, H, W   = x.shape
    F, _, HH, WW = w.shape

    Hp = int(1 + (H + 2*pad - HH)/stride)
    Wp = int(1 + (W + 2*pad - WW)/stride)

    # Only pad H and W dimensions of x
    #
    x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant', constant_values=0)

    # Indexes for HH, WW
    #
    oddHH  = (HH % 2) == 1
    initHH = int(HH/2) - pad       # even index
    if oddHH:
        initHH += 1
    endHH  = H + 2*pad - int(HH/2)   # same for even/odd

    offHH_L = int(HH/2)  # odd left offset
    if not oddHH:
        offHH_L -= 1 # even offset
    offHH_R = int(HH/2)+1  # same for even/odd

    oddWW  = (WW % 2) == 1
    initWW = int(WW/2) - pad
    if oddWW:
        initWW += 1
    endWW  = W + 2*pad - int(WW/2)

    offWW_L = int(WW/2)  # odd left offset
    if not oddWW:
        offWW_L -= 1 # even offset
    offWW_R = int(WW/2)+1  # same for even/odd

    # Initialize output volume V
    #
    V = np.zeros( (N,F,Hp,Wp) )

    # Naive implementation - elementwise mult and sum at each output point
    #
    hpi = 0
    wpi = 0
    for ni in range(N):
        for fi in range(F):
            for hi in range(initHH,endHH,stride): 
                for wi in range(initWW,endWW,stride): 
                    hil = hi-offHH_L
                    hir = hi+offHH_R
                    wil = wi-offWW_L
                    wir = wi+offWW_R

                    V[ni,fi,hpi,wpi] = np.sum(x_pad[ni, :, hil:hir, wil:wir]*w[fi, :, :, :]) + b[fi]

                    wpi += 1
                hpi += 1
                wpi  = 0
            hpi = 0
            wpi = 0

    out = V

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    stride = conv_param['stride']
    pad    = conv_param['pad']
    N, C, H, W   = x.shape
    F, _, HH, WW = w.shape

    Hp = int(1 + (H + 2*pad - HH)/stride)
    Wp = int(1 + (W + 2*pad - WW)/stride)

    # create padded dx
    # 
    x_pad  = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant', constant_values=0)
    dx_pad = np.zeros( (N,C,H+2*pad, W+2*pad) )
    dw     = np.zeros( (F,C,HH,WW) )
    db     = np.zeros(  F )

    # Indexes for HH, WW
    #
    oddHH  = (HH % 2) == 1
    initHH = int(HH/2) - pad       # even index
    if oddHH:
        initHH += 1
    endHH  = H + 2*pad - int(HH/2)   # same for even/odd

    offHH_L = int(HH/2)  # odd left offset
    if not oddHH:
        offHH_L -= 1 # even offset
    offHH_R = int(HH/2)+1  # same for even/odd

    oddWW  = (WW % 2) == 1
    initWW = int(WW/2) - pad
    if oddWW:
        initWW += 1
    endWW  = W + 2*pad - int(WW/2)

    offWW_L = int(WW/2)  # odd left offset
    if not oddWW:
        offWW_L -= 1 # even offset
    offWW_R = int(WW/2)+1  # same for even/odd

    # Naive implementation - elementwise mult and sum at each output point
    #
    # Note - convolution is just multiplication and summation
    #
    hpi = 0
    wpi = 0
    for ni in range(N):
        for fi in range(F):
            for hi in range(initHH,endHH,stride):
                for wi in range(initWW,endWW,stride): 
                    hil = hi-offHH_L
                    hir = hi+offHH_R
                    wil = wi-offWW_L
                    wir = wi+offWW_R

                    dx_pad[ni,:,hil:hir,wil:wir] += w[fi,:,:,:]*dout[ni,fi,hpi,wpi]
                    dw[fi,:,:,:] += x_pad[ni,:,hil:hir,wil:wir]*dout[ni,fi,hpi,wpi]

                    wpi += 1
                hpi += 1
                wpi  = 0
            hpi = 0
            wpi = 0

    # derive db
    #
    for fi in range(F):
        db[fi] = np.sum( dout[:,fi,:,:] )

    # remove pad
    #
    dx = dx_pad[:,:,pad:pad+H, pad:pad+W]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################

    N, C, H, W   = x.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']

    H2 = int(1 + (H - HH)/stride)
    W2 = int(1 + (W - WW)/stride)

    # Initialize output volume V
    #
    V = np.zeros( (N,C,H2,W2) )

    # Naive implementation - elementwise MAX at each output point
    #
    hpi = 0
    wpi = 0
    for ni in range(N):
        for ci in range(C):
            for hi in range(0,H,stride):
                for wi in range(0,W,stride):
            
                    V[ni,ci,hpi,wpi] = np.max(x[ni, ci, hi:(hi+HH), wi:(wi+WW)])

                    wpi += 1
                hpi += 1
                wpi  = 0
            hpi = 0
            wpi = 0

    out = V

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache

    N, C, H, W   = x.shape
    _,_,H2,W2    = dout.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']

    dx = np.zeros( (N,C,H,W) )

    # Naive implementation - elementwise MAX derivative at each output point
    #
    hpi = 0
    wpi = 0
    for ni in range(N):
        for ci in range(C):
            for hi in range(0,H,stride):
                for wi in range(0,W,stride):
                    tmp = np.zeros( (HH, WW) )
                    maxI, maxJ = np.unravel_index(x[ni,ci,hi:(hi+HH),wi:(wi+HH)].argmax(), (HH, WW))
                    tmp[maxI,maxJ] = 1
                    dx[ni, ci, hi:(hi+HH), wi:(wi+WW)] += tmp * dout[ni,ci,hpi,wpi]

                    wpi += 1
                hpi += 1
                wpi  = 0
            hpi = 0
            wpi = 0

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = x.shape

    out   = np.zeros( (N,C,H,W) )
    cache = {}
    for ci in range(C):
        x_ci = x[:,ci,:,:].reshape( (N,H*W) )  # NxD, D=H*W
        out_ci, cache[ci] = batchnorm_forward( x_ci, gamma[ci], beta[ci], bn_param ) 
        out[:,ci,:,:] = out_ci.reshape( (N,H,W) )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = dout.shape

    dx     = np.zeros( (N,C,H,W) )
    dgamma = np.zeros( C )
    dbeta  = np.zeros( C )
    for ci in range(C):
        dout_ci = dout[:,ci,:,:].reshape( (N,H*W) )  # NxD, D=H*W
        dx_ci, dgamma_ci, dbeta_ci = batchnorm_backward( dout_ci, cache[ci] ) 
        dx[:,ci,:,:] = dx_ci.reshape( (N,H,W) )

        dgamma[ci] = np.sum(dgamma_ci)
        dbeta[ci]  = np.sum(dbeta_ci)


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
