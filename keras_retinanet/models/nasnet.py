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
from keras.applications.nasnet import NASNetLarge
from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image
import keras.backend as k

class nasnetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return nasnet_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """
        BASE_WEIGHTS_PATH = ('https://github.com/titu1994/Keras-NASNet/'
                     'releases/download/v1.2/')
        NASNET_MOBILE_WEIGHT_PATH_NO_TOP = BASE_WEIGHTS_PATH + 'NASNet-mobile-no-top.h5'
        
        weights_path = keras.utils.get_file(
            'nasnet_mobile_no_top.h5',
            NASNET_MOBILE_WEIGHT_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='1ed92395b5b598bdda52abe5c0dbfd63')

        return weights_path

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['nasnet']
        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def nasnet_retinanet(num_classes, backbone='nasnet', inputs=None, modifier=None, **kwargs):

    k.clear_session()
    # choose default input
    if inputs is None:
        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(3, None, None))
        else:
            inputs = keras.layers.Input(shape=(None, None, 3))

    # create the resnet backbone
    if backbone == 'nasnet':
        nasnet_model = NASNetLarge(weights='imagenet', include_top=False, input_tensor=inputs)
    else:
        raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))

    # invoke modifier if given
    if modifier:
        nasnet_model = modifier(nasnet_model)
    concatenated_features=[nasnet_model.get_layer('add_3').output,
                           nasnet_model.get_layer('add_11').output,
                           nasnet_model.output]
    # create the full model
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=concatenated_features, **kwargs)
