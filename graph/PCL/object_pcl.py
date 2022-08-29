from torch import nn
from typing import Any, Optional
from torchvision.models._utils import IntermediateLayerGetter
try:
    from torchvision.models.utils import load_state_dict_from_url
except:
    from torchvision._internally_replaced_utils import load_state_dict_from_url
    # from torch.hub import load_state_dict_from_url
# from torchvision.models import mobilenetv3
from torchvision.models import resnet
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
# from torchvision.models.segmentation.fcn import FCN, FCNHead
# from torchvision.models.segmentation.lraspp import LRASPP
from collections import OrderedDict
from typing import Optional, Dict

from torch import nn, Tensor
from torch.nn import functional as F
from torch import nn
from torchvision.ops import roi_align



__all__ = ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101',
           'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large']


model_urls = {
    'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth',
    'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
    'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
    'deeplabv3_mobilenet_v3_large_coco':
        'https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth',
    'lraspp_mobilenet_v3_large_coco': 'https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth',
}


def _segm_model(
    name: str,
    backbone_name: str,
    num_classes: int,
    aux: Optional[bool],
    pretrained_backbone: bool = True
) -> nn.Module:
    if 'resnet' in backbone_name:
        backbone = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=[False, True, True])
        out_layer = 'layer4'
        out_inplanes = 2048
        aux_layer = 'layer3'
        aux_inplanes = 1024
    elif 'mobilenet_v3' in backbone_name:
        backbone = mobilenetv3.__dict__[backbone_name](pretrained=pretrained_backbone, dilated=True).features

        # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
        # The first and last blocks are always included because they are the C0 (conv1) and Cn.
        stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
        out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
        out_layer = str(out_pos)
        out_inplanes = backbone[out_pos].out_channels
        aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
        aux_layer = str(aux_pos)
        aux_inplanes = backbone[aux_pos].out_channels
    else:
        raise NotImplementedError('backbone {} is not supported as of now'.format(backbone_name))

    return_layers = {out_layer: 'out'}
    if aux:
        return_layers[aux_layer] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    classifier = model_map[name][0](out_inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model


def _load_model(
    arch_type: str,
    backbone: str,
    pretrained: bool,
    progress: bool,
    num_classes: int,
    aux_loss: Optional[bool],
    **kwargs: Any
) -> nn.Module:
    if pretrained:
        aux_loss = True
        kwargs["pretrained_backbone"] = False
    model = _segm_model(arch_type, backbone, num_classes, aux_loss, **kwargs)
    if pretrained:
        _load_weights(model, arch_type, backbone, progress)
    return model


def _load_weights(model: nn.Module, arch_type: str, backbone: str, progress: bool) -> None:
    arch = arch_type + '_' + backbone + '_coco'
    model_url = model_urls.get(arch, None)
    if model_url is None:
        raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
    else:
        state_dict = load_state_dict_from_url(model_url, progress=progress)
        model.load_state_dict(state_dict, strict=False)


def fcn_resnet50(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    **kwargs: Any
) -> nn.Module:
    """Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    """
    return _load_model('fcn', 'resnet50', pretrained, progress, num_classes, aux_loss, **kwargs)


def fcn_resnet101(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 32,
    aux_loss: Optional[bool] = None,
    **kwargs: Any
) -> nn.Module:
    """Constructs a Fully-Convolutional Network model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    """
    return _load_model('fcn', 'resnet101', pretrained, progress, num_classes, aux_loss, **kwargs)


def deeplabv3_resnet50(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    **kwargs: Any
) -> nn.Module:
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    """
    return _load_model('deeplabv3', 'resnet50', pretrained, progress, num_classes, aux_loss, **kwargs)


def deeplabv3_resnet101(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    **kwargs: Any
) -> nn.Module:
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): The number of classes
        aux_loss (bool): If True, include an auxiliary classifier
    """
    return _load_model('deeplabv3', 'resnet101', pretrained, progress, num_classes, aux_loss, **kwargs)


def deeplabv3_mobilenet_v3_large(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    **kwargs: Any
) -> nn.Module:
    """Constructs a DeepLabV3 model with a MobileNetV3-Large backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    """
    return _load_model('deeplabv3', 'mobilenet_v3_large', pretrained, progress, num_classes, aux_loss, **kwargs)

#
# def lraspp_mobilenet_v3_large(
#     pretrained: bool = False,
#     progress: bool = True,
#     num_classes: int = 21,
#     **kwargs: Any
# ) -> nn.Module:
#     """Constructs a Lite R-ASPP Network model with a MobileNetV3-Large backbone.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
#             contains the same classes as Pascal VOC
#         progress (bool): If True, displays a progress bar of the download to stderr
#         num_classes (int): number of output classes of the model (including the background)
#     """
#     if kwargs.pop("aux_loss", False):
#         raise NotImplementedError('This model does not use auxiliary loss')
#
#     backbone_name = 'mobilenet_v3_large'
#     if pretrained:
#         kwargs["pretrained_backbone"] = False
#     model = _segm_lraspp_mobilenetv3(backbone_name, num_classes, **kwargs)
#
#     if pretrained:
#         _load_weights(model, 'lraspp', backbone_name, progress)
#
#     return model
#


class _SimpleSegmentationModel(nn.Module):
    __constants__ = ['aux_classifier']
    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        aux_classifier: Optional[nn.Module] = None
    ) -> None:
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        # self.classifier = classifier
        self.classifier_ = classifier
        # self.aux_classifier = aux_classifier
        dim = classifier[-1].bias.shape[0]
        self.linear = nn.Linear(3*3*dim, dim) #64 * 2 * 2
        self.initialize()

    def forward(self, x, object, object_mask) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        B, C, H, W = x.shape
        # contract: features is a dict of tensors
        features = self.backbone(x)

        # result = OrderedDict()
        x = features["out"]
        x = self.classifier_(x)
        # x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        # object = object.reshape(-1, 5)[object_mask.reshape(-1)==1]
        object_new = object.clone()
        if object.max() <= 1.:
            object_new[:, :, 1] = object[:, :, 1] * W
            object_new[:, :, 2] = object[:, :, 2] * H
            object_new[:, :, 3] = object[:, :, 3] * W
            object_new[:, :, 4] = object[:, :, 4] * H
        for i in range(len(object_new)):
            object_new[i, ..., 0] = i
        rois_feature = roi_align(x, object_new.reshape(-1, 5), (3, 3), 1.0/(2**3), aligned=True).reshape(object_new.shape[0], object_new.shape[1], -1)
        rois_feature = self.linear(rois_feature)
        rois_feature = F.normalize(rois_feature, dim=-1)
        return rois_feature

    def initialize(self):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 and classname.find("BasicConv") == -1:
                m.weight.data.normal_(0.0, 0.02)
                try:
                    m.bias.data.fill_(0.001)
                except:
                    pass
                # print("Initialized {}".format(m))
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
                # print("Initialized {}".format(m))
            elif classname.find('Linear') != -1:
                m.weight.data.normal_(0.0, 0.02)
                try:
                    m.bias.data.fill_(0.001)
                except:
                    pass
                # print("Initialized {}".format(m))

        self.backbone.apply(weights_init)
        self.classifier_.apply(weights_init)
        self.linear.apply(weights_init)

class FCN(_SimpleSegmentationModel):
    """
    Implements a Fully-Convolutional Network for semantic segmentation.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)

