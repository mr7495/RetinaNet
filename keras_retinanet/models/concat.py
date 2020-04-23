import keras
from keras.utils import get_file
from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image
import keras.backend as k

class concatBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return concat_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        Weights can be downloaded at https://github.com/fizyr/keras-models/releases .
        """
        resource = keras.applications.vgg16.vgg16.WEIGHTS_PATH_NO_TOP
        checksum = '6d6bbae143d832006294945121d1f1fc'

        return get_file(
            '{}_weights_tf_dim_ordering_tf_kernels_notop.h5'.format(self.backbone),
            resource,
            cache_subdir='models',
            file_hash=checksum
        )

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['concat']

        if self.backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def concat_retinanet(num_classes, backbone='concat', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a vgg backbone.
    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('vgg16', 'vgg19')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).
    Returns
        RetinaNet model with a VGG backbone.
    """
    k.clear_session()
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))

    # create the vgg backbone
    if backbone == 'concat':
        vgg = keras.applications.VGG16(input_tensor=inputs, include_top=False, weights='imagenet')
        resnet=keras.applications.resnet.ResNet50(input_tensor=inputs, include_top=False, weights='imagenet')
        
    else:
        raise ValueError("Backbone '{}' not recognized.".format(backbone))

    if modifier:
        vgg = modifier(vgg)
        resnet = modifier(resnet)

    # create the full model
    vgg_layer_names = ["block3_pool", "block4_pool", "block5_pool"]
    resnet_layer_names=["conv3_block4_out","conv4_block6_out","conv5_block3_out"]
    layer_outputs = [vgg.get_layer(name).output for name in vgg_layer_names]
    layer_outputs2 = [resnet.get_layer(name).output for name in resnet_layer_names]
    layer_outputs.extend(layer_outputs2)
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs, **kwargs)
