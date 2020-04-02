import tensorflow as tf
# -------------------------------------------------------------------
# conv4d equivalent with dilation
# -------------------------------------------------------------------


def convolve4d(input, filter, 
           strides=[1, 1, 1, 1, 1, 1],
           padding='SAME',
           dilation_rate=None,
           stack_axis=None,
           stack_nested=False,
          ):
  '''
    Computes a convolution over 4 dimensions.
    Python generalization of tensorflow's conv3d with dilation.
    conv4d_stacked uses tensorflows conv3d and stacks results along
    stack_axis.

Parameters
----------
input : A Tensor.
        Shape [batch, x_dim, y_dim, z_dim, t_dim, in_channels]

filter: A Tensor. Must have the same type as input. 
        Shape [x_dim, y_dim, z_dim, t_dim, in_channels, out_channels]. 
        in_channels must match between input and filter

strides: A list of ints that has length 6. 1-D tensor of length 6. 
         The stride of the sliding window for each dimension of input. 
         Must have strides[0] = strides[5] = 1.
padding: A string from: "SAME", "VALID". The type of padding algorithm to use.

dilation_rate: Optional. Sequence of 4 ints >= 1. 
               Specifies the filter upsampling/input downsampling rate. 
               Equivalent to dilation_rate in tensorflows tf.nn.convolution

stack_axis: Int
          Axis along which the convolutions will be stacked.
          By default the axis with the lowest output dimensionality will be 
          chosen. This is only an educated guess of the best choice!

stack_nested: Bool
          If set to True, this will stack in a for loop seperately and afterwards 
          combine the results. In most cases slower, but maybe less memory needed.
      
Returns
-------
        A Tensor. Has the same type as input.
  '''

  stack_axis = 3

  if dilation_rate != None:
    dilation_along_stack_axis = dilation_rate[stack_axis-1]
  else:
    dilation_along_stack_axis = 1

  tensors_t = tf.unstack(input,axis=stack_axis)
  kernel_t = tf.unstack(filter,axis=stack_axis-1)

  noOfInChannels = input.get_shape().as_list()[-1]
  len_ts = filter.get_shape().as_list()[stack_axis-1]
  size_of_t_dim = input.get_shape().as_list()[stack_axis]

  if len_ts % 2 ==1:
    # uneven filter size: same size to left and right
    filter_l = int(len_ts/2)
    filter_r = int(len_ts/2)
  else:
    # even filter size: one more to right
    filter_l = int(len_ts/2) -1
    filter_r = int(len_ts/2)

  # The start index is important for strides and dilation
  # The strides start with the first element
  # that works and is VALID:
  start_index = 0
  if padding == 'VALID':
    for i in range(size_of_t_dim):
      if len( range(  max(i - dilation_along_stack_axis*filter_l,0), 
                      min(i + dilation_along_stack_axis*filter_r+1,
                        size_of_t_dim),dilation_along_stack_axis)
            ) == len_ts:
        # we found the first index that doesn't need padding
        break
    start_index = i
    # print 'start_index', start_index

  # loop over all t_j in t
  result_t = []
  for i in range(start_index, size_of_t_dim, strides[stack_axis]):

      kernel_patch = []
      input_patch = []
      tensors_t_convoluted = []

      if padding == 'VALID':

        # Get indices t_s
        indices_t_s = range(  max(i - dilation_along_stack_axis*filter_l, 0),
                              min(i + dilation_along_stack_axis*filter_r+1, size_of_t_dim),
                              dilation_along_stack_axis)

        # check if Padding = 'VALID'
        if len(indices_t_s) == len_ts:

          # sum over all remaining index_t_i in indices_t_s
          for j,index_t_i in enumerate(indices_t_s):
            if not stack_nested:
              kernel_patch.append(kernel_t[j])
              input_patch.append(tensors_t[index_t_i])
            else:
              if dilation_rate != None:
                tensors_t_convoluted.append( tf.nn.convolution(input=tensors_t[index_t_i],
                                               filter=kernel_t[j],
                                               strides=strides[1:stack_axis+1]+strides[stack_axis:5],
                                               padding=padding,
                                               dilation_rate=dilation_rate[:stack_axis-1]+dilation_rate[stack_axis:])
                                          )
              else:
                  tensors_t_convoluted.append( tf.nn.conv3d(input=tensors_t[index_t_i],
                                                   filter=kernel_t[j],
                                                   strides=strides[:stack_axis]+strides[stack_axis+1:],
                                                   padding=padding)
                                              )
          if stack_nested:
            sum_tensors_t_s = tf.add_n(tensors_t_convoluted)
            # put together
            result_t.append(sum_tensors_t_s)

      elif padding == 'SAME':

        # Get indices t_s
        indices_t_s = range(i - dilation_along_stack_axis*filter_l, 
                            (i + 1) + dilation_along_stack_axis*filter_r,
                            dilation_along_stack_axis)

        for kernel_j,j in enumerate(indices_t_s):
          # we can just leave out the invalid t coordinates
          # since they will be padded with 0's and therfore
          # don't contribute to the sum

          if 0 <= j < size_of_t_dim:
            if not stack_nested:
              kernel_patch.append(kernel_t[kernel_j])
              input_patch.append(tensors_t[j])
            else:
              if dilation_rate != None:
                  tensors_t_convoluted.append( tf.nn.convolution(input=tensors_t[j],
                                                 filter=kernel_t[kernel_j],
                                                 strides=strides[1:stack_axis+1]+strides[stack_axis:5],
                                                 padding=padding,
                                                 dilation_rate=dilation_rate[:stack_axis-1]+dilation_rate[stack_axis:])
                                            )
              else:
                  tensors_t_convoluted.append( tf.nn.conv3d(input=tensors_t[j],
                                                     filter=kernel_t[kernel_j],
                                                     strides=strides[:stack_axis]+strides[stack_axis+1:],
                                                     padding=padding)
                                                )
        if stack_nested:
          sum_tensors_t_s = tf.add_n(tensors_t_convoluted)
          # put together
          result_t.append(sum_tensors_t_s)

      if not stack_nested:
        if kernel_patch:
          kernel_patch = tf.concat(kernel_patch,axis=3)
          input_patch = tf.concat(input_patch,axis=4)
          if dilation_rate != None:
              result_patch = tf.nn.convolution(input=input_patch,
                                         filter=kernel_patch,
                                         strides=strides[1:stack_axis]+strides[stack_axis+1:5],
                                         padding=padding,
                                         dilation_rate=dilation_rate[:stack_axis-1]+dilation_rate[stack_axis:])
          else:
              result_patch = tf.nn.conv3d(input=input_patch,
                                         filter=kernel_patch,
                                         strides=strides[:stack_axis]+strides[stack_axis+1:],
                                         padding=padding)
          result_t.append(result_patch)

  # stack together
  return tf.stack(result_t,axis=stack_axis)