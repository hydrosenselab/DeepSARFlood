import timm
import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable, Optional
import abc

class Encoder(nn.Module):
    def __init__(self,output_stride=None):
        super().__init__()
        self.model = timm.create_model('resnet50', pretrained=True,in_chans=6,features_only=True,out_indices=[0,1,2,3,4])
        self.output_stride = 32 if output_stride is None else output_stride
        self.out_channels = [6] + self.model.feature_info.channels()
    def forward(self, x):
        features = self.model(x)
        features = [x] + features
        return features

class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()

        super().__init__(conv, bn, relu)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError(f"Attention {name} is not implemented")

    def forward(self, x):
        return self.attention(x)


def initialize_decoder(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetPlusPlusDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            err = f"Model depth is {n_blocks}, but you provide `decoder_channels` for {len(decoder_channels)} blocks."  # noqa: E501
            raise ValueError(err)

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels
        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (
                        layer_idx + 1 - depth_idx
                    )
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(
                    in_ch, skip_ch, out_ch, **kwargs
                )
        blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1], **kwargs
        )
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, *features):
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](
                        features[depth_idx], features[depth_idx + 1]
                    )
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [
                        dense_x[f"x_{idx}_{dense_l_i}"]
                        for idx in range(depth_idx + 1, dense_l_i + 1)
                    ]
                    cat_features = torch.cat(
                        cat_features + [features[dense_l_i + 1]], dim=1
                    )
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[
                        f"x_{depth_idx}_{dense_l_i}"
                    ](dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features)
        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](
            dense_x[f"x_{0}_{self.depth-1}"]
        )
        return dense_x[f"x_{0}_{self.depth}"]
    
class SegmentationHead(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        activation=nn.Identity(),
        upsampling=1,
    ):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        super().__init__(conv2d, upsampling, activation)

class SegmentationModel(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)

    def check_input_shape(self, x):
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (
                (h // output_stride + 1) * output_stride
                if h % output_stride != 0
                else h
            )
            new_w = (
                (w // output_stride + 1) * output_stride
                if w % output_stride != 0
                else w
            )
            err = f"""
                Wrong input shape height={h}, width={w}. Expected image height
                and width divisible by {output_stride}. Consider pad your images
                to shape ({new_h}, {new_w}).
            """
            raise RuntimeError(err)

    def forward(self, x):
        """Sequentially pass `x` through model`s encoder, decoder and heads"""
        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)
        return masks


class UnetPlusPlus(SegmentationModel):
    def __init__(
        self,
        encoder_depth: int = 5,
        decoder_use_batchnorm: bool = True,
        decoder_channels: list[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        classes: int = 2,
        activation: Callable = nn.Identity(),
    ):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=False ,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,)