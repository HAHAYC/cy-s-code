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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint
import moxing.mxnet as mox
import mxnet as mx
from moxing.mxnet.module.rcnn.tester import Predictor, pred_eval
from moxing.mxnet.config.rcnn_config import config
# set environment parameters
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['PYTHONUNBUFFERED'] = '1'

def e2e_eval(args, image_list_eval):
    # default
    assert args.train_url != None, 'checkpoint_url should not be None'
    assert args.load_epoch != None, 'load_epoch should not be None'

    config.TEST.HAS_RPN = True
    pprint.pprint(mox.rcnn_config.config)
    # load test data
    test_data, imdb = mox.get_data_iter('rcnn_eval', image_set_list=image_list_eval,
                                        args=args, worker_id=mox.get_hyper_parameter('worker_id'))
    # load symbol
    symbol = mox.get_model('object_detection', 'resnet_rcnn', num_classes=args.num_classes, is_train=False)
    # load model params
    model_prefix = os.path.join(args.train_url, 'fine_tune')
    load_epoch = args.load_epoch
    arg_params, aux_params = mox.rcnn_load_param(
        model_prefix, load_epoch,
        convert=True, data=test_data, process=True,
        is_train=False, sym=symbol)
    max_data_shape = [('data', (1, 3, max([v[0] for v in mox.rcnn_config.config.SCALES]),
        max([v[1] for v in mox.rcnn_config.config.SCALES])))]
    # create predictor
    devs = [mx.gpu(0)]
    predictor = Predictor(
        symbol,
        data_names=[k[0] for k in test_data.provide_data],
        label_names=None,
        context=devs,
        max_data_shapes=max_data_shape,
        provide_data=test_data.provide_data,
        provide_label=test_data.provide_label,
        arg_params=arg_params,
        aux_params=aux_params)

    # start detection
    pred_eval(predictor, test_data, imdb, vis=False, thresh=0.001)

