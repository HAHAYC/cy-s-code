from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
logging.basicConfig(level=logging.DEBUG)
import os
import moxing.mxnet as mox
import argparse
import fine_tune_config
import numpy as np
import logging
import mxnet as mx
from eval_faster_rcnn import e2e_eval

def rcnn_do_eval(prefix, means, stds, argparse, image_list_eval):
    """
    saves a model checkpoint for rcnn and do evaluation.

    :param prefix: Prefix of model name.
    :param means: means
    :param stds:  stds
    :param args:  param evaluation need
    :return: callback function.
    """
    def _callback(iter_no, sym, arg, aux):
        # save the checkpoint
        if arg.get('bbox_pred_weight') is not None:
            arg['bbox_pred_weight_test'] = \
                (arg['bbox_pred_weight'].T * mx.nd.array(stds)).T
            arg['bbox_pred_bias_test'] = \
                arg['bbox_pred_bias'] * mx.nd.array(stds) + mx.nd.array(means)
            mx.model.save_checkpoint(os.path.join(prefix, 'fine_tune'), iter_no + 1, sym, arg, aux)
            arg.pop('bbox_pred_weight_test')
            arg.pop('bbox_pred_bias_test')
        if arg.get('rfcn_bbox_weight') is not None:
            arg['rfcn_bbox_weight_test'] = \
                (arg['rfcn_bbox_weight'].T * mx.nd.array(stds)).T
            arg['rfcn_bbox_bias_test'] = \
                arg['rfcn_bbox_bias'] * mx.nd.array(stds) + mx.nd.array(means)
            mx.model.save_checkpoint(os.path.join(prefix, 'fine_tune'), iter_no + 1, sym, arg, aux)
            arg.pop('rfcn_bbox_weight_test')
            arg.pop('rfcn_bbox_bias_test')
        # do evaluation
        if (iter_no + 1) % argparse.eval_frequence is 0:
            if argparse.train_url is not None and len(argparse.train_url):
                argparse.load_epoch = iter_no + 1
                if mox.get_hyper_parameter('worker_id') is 0:
                    e2e_eval(args=argparse, image_list_eval=image_list_eval)
    return _callback

def add_parameter():
    parser = argparse.ArgumentParser(description='train faster rcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_url', type=str, help='the pre-trained model')
    parser.add_argument('--train_url', type=str, help='the path model saved')
    parser.add_argument('--num_classes', type=int, help='the number of classes')
    parser.add_argument('--load_epoch', type=int, help='load the model on epoch use checkpoint_url')
    parser.add_argument('--num_epoch', type=int, help='the number of training epochs')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=0.0005)
    parser.add_argument('--network', type=str, default='resnet_rcnn', help='name of network')
    parser.add_argument('--labels_name_url', type=str, default=None, help='the classes txt file name')
    parser.add_argument('--eval_frequence', type=int, default=1, help='frequence of evaluation')
    parser.add_argument('--export_model', type=bool, default=True, help='change train url to model,metric.json')
    parser.add_argument('--split_spec', type=int, default=0.8, help='percent when split data to train and evaluation')
    args, _ = parser.parse_known_args()
    return args

# get fine tune config
def set_config(args):
    mox.rcnn_config.config = fine_tune_config.config
    mox.rcnn_config.default = fine_tune_config.default
    if args.checkpoint_url is not None:
        mox.rcnn_config.default.pretrained = args.checkpoint_url
    if args.load_epoch is not None:
        mox.rcnn_config.default.pretrained_epoch = args.load_epoch
    if args.num_classes is not None:
        mox.rcnn_config.config.NUM_CLASSES = args.num_classes
    if args.lr is not None:
        mox.rcnn_config.default.base_lr = args.lr
    if args.train_url is not None:
        mox.rcnn_config.default.e2e_prefix = args.train_url
    if args.num_epoch is not None:
        mox.rcnn_config.default.e2e_epoch = args.num_epoch
    mox.rcnn_config.default.dataset_path = mox.get_hyper_parameter('data_url')

def get_model(data_set, symbol, ctx):
    input_batch_size = data_set.batch_size
    max_data_shape = [('data', (input_batch_size, 3, max([v[0] for v in mox.rcnn_config.config.SCALES]),
                                max([v[1] for v in mox.rcnn_config.config.SCALES])))]
    max_data_shape, max_label_shape = data_set.infer_shape(max_data_shape)
    max_data_shape.append(('gt_boxes', (input_batch_size, 100, 5)))
    logger = logging.getLogger()
    logger.info('providing maximum shape %s %s' % (max_data_shape, max_label_shape))
    model = mox.MutableModule(
        symbol,
        data_names=[k[0] for k in data_set.provide_data],
        label_names=[k[0] for k in data_set.provide_label],
        logger=logger, context=ctx, work_load_list=None,
        max_data_shapes=max_data_shape, max_label_shapes=max_label_shape,
        fixed_param_prefix=mox.rcnn_config.config.FIXED_PARAMS)
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    os.environ['PYTHONUNBUFFERED'] = '1'
    return model

def train_faster_rcnn():
    args = add_parameter()
    data_path = mox.get_hyper_parameter('data_url')
    # find if train file is only exist and spilt it to train and eval
    mox.file.set_auth(path_style=True)
    mox.set_hyper_parameter('data_url', data_path)
    set_config(args)
    num_gpus = mox.get_hyper_parameter('num_gpus')
    ctx = [mx.cpu()] if num_gpus is None or num_gpus == 0 else [mx.gpu(int(i)) for i in range(num_gpus)]
    num_classes = mox.rcnn_config.config.NUM_CLASSES

    # set train parameters
    image_list_train, image_list_eval = mox.get_image_list(data_path, args.split_spec)
    symbol = mox.get_model('object_detection', args.network, num_classes=num_classes)
    data_set = mox.get_data_iter('rcnn_train', image_set_list=image_list_train, sym=symbol, ctx=ctx, args=args)
    num_examples = len(data_set.roidb)
    batch_size = data_set.batch_size
    initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)
    metrics = [mox.contrib_metrics.RPNAccMetric(),
               mox.contrib_metrics.RPNLogLossMetric(),
               mox.contrib_metrics.RPNL1LossMetric(),
               mox.contrib_metrics.RCNNAccMetric(),
               mox.contrib_metrics.RCNNLogLossMetric(),
               mox.contrib_metrics.RCNNL1LossMetric(),
               mx.metric.CompositeEvalMetric()]
    if 'rfcn' in args.network:
        means = np.tile(np.array(mox.rcnn_config.config.TRAIN.BBOX_MEANS), num_classes * 7 * 7)
        stds = np.tile(np.array(mox.rcnn_config.config.TRAIN.BBOX_STDS), num_classes * 7 * 7)
    else:
        means = np.tile(np.array(mox.rcnn_config.config.TRAIN.BBOX_MEANS), num_classes)
        stds = np.tile(np.array(mox.rcnn_config.config.TRAIN.BBOX_STDS), num_classes)

    # set end callback evaluation and create metric.json
    if args.train_url is not None and len(args.train_url):
        worker_id = mox.get_hyper_parameter('worker_id')
        save_path = args.train_url if worker_id == 0 else "%s-%d" % (args.train_url, worker_id)
        epoch_end_callbacks = rcnn_do_eval(save_path, means, stds, args, image_list_eval=image_list_eval)
    else:
        epoch_end_callbacks = None

    params_tuple = mox.rcnn_load_param(
        prefix=mox.rcnn_config.default.pretrained,
        epoch=mox.rcnn_config.default.pretrained_epoch,
        convert=True, data=data_set, sym=symbol, is_train=True)
    lr = mox.rcnn_config.default.base_lr
    lr_factor = 0.1
    lr_iters = [int(epoch * num_examples / batch_size)
                for epoch in [int(i) for i in mox.rcnn_config.default.e2e_lr_step.split(',')]]
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)
    optimizer_params = {'momentum': args.mom,
                        'wd': args.wd,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': (1.0 / batch_size),
                        'clip_gradient': 5}
    # mox run
    mox.run(data_set=(data_set, None),
            optimizer=args.optimizer,
            optimizer_params=optimizer_params,
            run_mode=mox.ModeKeys.TRAIN,
            model=get_model(data_set, symbol, ctx),
            epoch_end_callbacks=epoch_end_callbacks,
            initializer=initializer,
            batch_size=batch_size,
            params_tuple=params_tuple,
            metrics=metrics,
            num_epoch=mox.rcnn_config.default.e2e_epoch)

    if mox.get_hyper_parameter('worker_id') is 0:
        # remove the temp file
        data_path = mox.get_hyper_parameter('data_url')
        mox.file.remove(os.path.join(data_path, 'cache'), recursive=True)
        if args.export_model:
            # change train url to {model, metric.json}
            mox.export_model(args.train_url)

if __name__ == '__main__':
    train_faster_rcnn()
