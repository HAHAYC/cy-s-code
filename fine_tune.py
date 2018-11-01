# Copyright 2018 Deep Learning Service of Huawei Cloud. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys
sys.path.insert(0, '/home/work/user-job-dir/moxing/')
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
import mxnet as mx
import moxing.mxnet as mox

def get_fine_tune_model(symbol, arg_params, num_classes, layer_name):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k: arg_params[k] for k in arg_params if 'fc' not in k})
    return net, new_args

def get_data():
    train_file = mox.get_hyper_parameter('train_file')
    val_file = mox.get_hyper_parameter('val_file')
    data_url = mox.get_hyper_parameter('data_url')
    train_path = os.path.join(data_url, train_file)
    val_path = os.path.join(data_url, val_file)
    train_list, val_list = mox.get_image_list(train_path, 0.8)
    image_shape = args.image_shape.split(',')
    if mox.file.is_directory(train_path) and mox.file.is_directory(val_path):
        data_set = mox.get_data_iter('imageraw',
                                     hyper_train={'data_shape': tuple(int(i) for i in image_shape), 'batch_size': args.batch_size},
                                     hyper_val={'data_shape': tuple(int(i) for i in image_shape), 'batch_size': args.batch_size},
                                     num_process=128, train_img_list=train_list, val_img_list=val_list)
    else:
        data_set = mox.get_data_iter('imagerecord',
                                     hyper_train={'data_shape': tuple(int(i) for i in image_shape), 'batch_size': args.batch_size},
                                     hyper_val={'data_shape': tuple(int(i) for i in image_shape), 'batch_size': args.batch_size})
    return data_set

def get_classname():
    train_file = mox.get_hyper_parameter('train_file')
    val_file = mox.get_hyper_parameter('val_file')
    data_url = mox.get_hyper_parameter('data_url')
    train_path = os.path.join(data_url, train_file)
    classname_list = None
    if mox.file.is_directory(train_path):
        classname_list = mox.file.list_directory(train_path)
    return classname_list

def get_optimizer_params():
    optimizer_params = mox.get_optimizer_params(num_examples=args.num_examples, lr=args.lr,
                                                batch_size=args.batch_size)
    return optimizer_params

def get_model(network):
    num_gpus = mox.get_hyper_parameter('num_gpus')
    devs = mx.cpu() if num_gpus is None or num_gpus == 0 else [mx.gpu(int(i)) for i in range(num_gpus)]
    model = mx.mod.Module(
        context       = devs,
        symbol        = network
    )
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '1'
    return model

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="fine-tune a dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_url', type=str, help='the pre-trained model path')
    parser.add_argument('--train_url', type=str, help='the path model saved')
    parser.add_argument('--layer_before_fullc', type=str, default='flatten0',
                        help='the name of the layer before the last fullc layer')
    parser.add_argument('--num_classes', type=int, help='the number of classes')
    parser.add_argument('--num_examples', type=int, default=7370, help='the number of training examples')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--load_epoch', type=int, default=0)
    parser.add_argument('--num_epoch', type=int, default=1, help='the number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lr_factor', type=float, default=0.1,
                        help='the ratio to reduce lr on each step')
    parser.add_argument('--lr_step_epochs', type=str,
                        help='the epochs to reduce the lr, e.g. 30,60')
    parser.add_argument('--save_frequency', type=int, default=1, help='how many epochs to save model')
    parser.add_argument('--split_spec', type=int, default=0.8, help='split percent of train and eval')
    parser.add_argument('--export_model', type=bool, default=True, help='change train url to model,metric.json')
    parser.add_argument('--image_shape', type=str, default='3,224,224', help='images shape')
    args, _ = parser.parse_known_args()
    mox.set_hyper_parameter('disp_batches', 10)
    # load data
    if not mox.file.exists(os.path.join(mox.get_hyper_parameter('data_url'), 'train.rec'))\
      and not mox.file.exists(os.path.join(mox.get_hyper_parameter('data_url'), 'val.rec')):
        data_url = mox.get_hyper_parameter('data_url')
        file_name = data_url.split('/')[-2]
        mox.set_hyper_parameter('train_file', file_name)
        mox.set_hyper_parameter('val_file', file_name)
        mox.set_hyper_parameter('data_url', data_url[:-len(file_name)-1])
    data_set = get_data()

    # get total labels
    labels = []
    total_label = []
    metrics = [mx.metric.Accuracy(), mx.metric.CrossEntropy()]
    for i in range(args.num_classes):
        labels.append(str(i))
    else:
        total_label = get_classname()
    if total_label is None:
        total_label = labels
    if args.train_url is not None and len(args.train_url) != 0:
        metrics = mox.contrib_metrics.GetMetricsmulticlass(labels=labels, total_label=total_label,
                                                           train_url=args.train_url)

    # load pretrained model
    sym, arg_params, aux_params = mox.load_model(args.checkpoint_url, args.load_epoch)

    # remove the last fullc layer
    (new_sym, new_args) = get_fine_tune_model(
        sym, arg_params, args.num_classes, args.layer_before_fullc)

    #load model
    model = get_model(new_sym)
    params_tuple = (new_args, aux_params)
    if args.train_url is not None:
        worker_id = mox.get_hyper_parameter('worker_id')
        save_path = args.train_url if worker_id == 0 else "%s-%d" % (args.train_url, worker_id)
        epoch_end_callbacks = mx.callback.do_checkpoint(save_path, args.save_frequency)
    else:
        epoch_end_callbacks = None

    # train
    mox.run(data_set, model, params_tuple, run_mode=mox.ModeKeys.TRAIN, optimizer='sgd',
            optimizer_params=get_optimizer_params(), batch_size=args.batch_size,
            epoch_end_callbacks=epoch_end_callbacks, num_epoch=args.num_epoch,
            load_epoch=args.load_epoch, metrics=metrics,)

    # remove temp file and sort out result
    if mox.get_hyper_parameter('worker_id') is 0:
        # remove the temp file
        data_path = mox.get_hyper_parameter('data_url')
        if mox.file.exists(os.path.join(data_path, 'cache')):
            mox.file.remove(os.path.join(data_path, 'cache'), recursive=True)
        if args.export_model:
            # change train url to {model, metric.json}
            mox.export_model(args.train_url)
