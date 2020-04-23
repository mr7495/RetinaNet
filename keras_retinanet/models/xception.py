"""
Copyright 2017-2018 Fizyr (https://fizyr.com)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
from keras.utils import get_file
from keras.applications.xception import Xception
from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image
import keras.backend as k

class XceptionBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return xception_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """
        TF_WEIGHTS_PATH_NO_TOP = (
            'https://github.com/fchollet/deep-learning-models/'
            'releases/download/v0.4/'
            'xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
        weights_path = keras.utils.get_file(
            'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
            TF_WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='b0042744bf5b25fce3cb969f33bebb97')

        return weights_path

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['xception']
        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def xception_retinanet(num_classes, backbone='xception', inputs=None, modifier=None, **kwargs):

    k.clear_session()
    # choose default input
    if inputs is None:
        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(3, None, None))
        else:
            inputs = keras.layers.Input(shape=(None, None, 3))

    # create the resnet backbone
    if backbone == 'xception':
        xception_model = Xception(weights='None', include_top=False, input_tensor=inputs)
    else:
        raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))

    # invoke modifier if given
    if modifier:
        xception_model = modifier(xception_model)
    concatenated_features=[xception_model.get_layer('block4_sepconv1_act').output,
                           xception_model.get_layer('add_12').output,
                           xception_model.output]
    # create the full model
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=concatenated_features, **kwargs)


