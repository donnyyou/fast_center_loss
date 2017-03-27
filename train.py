import mxnet as mx
import numpy as np
from center_loss import *
from prefetch_data_iter import PrefetchDataIter
# from data import mnist_iterator
import logging
import train_model
import argparse

parser = argparse.ArgumentParser(description='train mnist use softmax and centerloss')
parser.add_argument('--gpus', type=str, default='',
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--batch-size', type=int, default=256,
                    help='the batch size')
parser.add_argument('--num-examples', type=int, default=60000,
                    help='the number of training examples')
parser.add_argument('--lr', type=float, default=.05,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=0.5,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=float, default=20,
                    help='the number of epoch to factor the lr, could be .5')
parser.add_argument('--model-prefix', type=str,
                    help='the prefix of the model to load')
parser.add_argument('--save-model-prefix', type=str,default='./model/center_loss',
                    help='the prefix of the model to save')
parser.add_argument('--num-epochs', type=int, default=100,
                    help='the number of training epochs')
parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')
parser.add_argument('--log_file', type=str, default='log2.txt',
                    help='log file')
parser.add_argument('--log_dir', type=str, default='.',
                    help='log dir')
args = parser.parse_args()


data_shape = (3,112,96)

def resnet_block_fast(input_data, num_filter, name):
    conv1 = mx.symbol.Convolution(data=input_data, 
                                     kernel=(3, 3), 
                                     stride=(2, 2), 
                                     pad=(0, 0), 
                                     num_filter=num_filter, 
                                     name='conv'+name)
    conv1 = mx.symbol.Activation(data=conv1, act_type='relu', name='relu'+name)
    conv2 = mx.symbol.Convolution(data=conv1, 
                                     kernel=(3, 3), 
                                     stride=(1, 1), 
                                     pad=(1, 1), 
                                     num_filter=num_filter, 
                                     name='conv'+name+'_1')
    conv2 = mx.symbol.Activation(data=conv2, act_type='relu', name='relu'+name+'_1')
    conv3 = mx.symbol.Convolution(data=conv2, 
                                     kernel=(3, 3), 
                                     stride=(1, 1), 
                                     pad=(1, 1), 
                                     num_filter=num_filter, 
                                     name='conv'+name+'_2')
    conv3 = mx.symbol.Activation(data=conv3, act_type='relu', name='relu'+name+'_2')
    return conv3 + conv1

def get_symbol(batchsize=64):
    """
    LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
    Haffner. "Gradient-based learning applied to document recognition."
    Proceedings of the IEEE (1998)
    """
    data = mx.symbol.Variable('data')

    softmax_label = mx.symbol.Variable('softmax_label')
    center_label = mx.symbol.Variable('center_label')

    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), 
                                     stride=(2, 2), pad=(0, 0), 
                                     num_filter=32, name='conv1')
    conv1 = mx.symbol.Activation(data=conv1, act_type='relu', name='relu1')
    res2 = resnet_block_fast(conv1, 32, '2')
    res3 = resnet_block_fast(res2, 32, '3')
    res4 = resnet_block_fast(res3, 64, '4')
    res5 = resnet_block_fast(res4, 128, '5')
    flatten = mx.symbol.Flatten(data=res5)
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=256, name='fc6')
    fc7 = mx.symbol.FullyConnected(data=fc6, num_hidden=10575, name='fc7')

    ce_loss = mx.symbol.SoftmaxOutput(data=fc7, label=softmax_label, name='softmax')

    center_loss_ = mx.symbol.Custom(data=fc6, label=center_label, name='center_loss_', op_type='centerloss',\
            num_class=10575, alpha=0.5, scale=0.008, batchsize=batchsize)
    center_loss = mx.symbol.MakeLoss(name='center_loss', data=center_loss_)
    mlp = mx.symbol.Group([ce_loss, center_loss])

    return mlp

def main():
    batchsize = args.batch_size if args.gpus is '' else \
        args.batch_size / len(args.gpus.split(','))
    print 'batchsize is ', batchsize

    # define network structure
    net = get_symbol(batchsize)

    # load data
    train_img_list = './data/train_correct.txt'
    val_img_list = './data/test_correct.txt'

    train = PrefetchDataIter(train_img_list, batch_size = args.batch_size, is_color = True, root_dir="/home/donny/112x96/")
    val = PrefetchDataIter(val_img_list, batch_size = args.batch_size, is_color = True, root_dir="/home/donny/112x96/")
    # train, val = mnist_iterator(batch_size=args.batch_size, input_shape=data_shape)

    # train
    # ctx = mx.gpu(1)
    # mod = mx.mod.Module(net, context = ctx, data_names = ('data',), label_names = ('softmax_label', 'center_label',))
    # mod.bind(data_shapes=train.provide_data,
    #         label_shapes=train.provide_label)

    # mod.fit(train, eval_data=val,
    #        optimizer_params={'learning_rate':0.01, 'momentum': 0.9}, num_epoch=30)
    print 'training model ...'
    train_model.fit(args, net, (train, val), data_shape)

if __name__ == "__main__":
    main()
