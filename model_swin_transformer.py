import torch
from torch import nn
import torch.nn.functional as F
import timm
import abc
from einops import rearrange


class Encoder(nn.Module):
    def __init__(self,output_stride=None):
        super().__init__()
        self.model = timm.create_model("swinv2_base_window8_256", pretrained=False,in_chans=6,features_only=True,out_indices=[0,1,2,3],img_size=512)
        self.output_stride = 32 if output_stride is None else output_stride
        self.out_channels = [6] + self.model.feature_info.channels()
    def forward(self, x):
        features = self.model(x)
        features = [x] + features
        for i in range(len(features)):
            if (
                not features[i].shape[1] == self.out_channels[i]
                and features[i].shape[-1] == self.out_channels[i]
            ):
                features[i] = rearrange(features[i], "b h w c -> b c h w")
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


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
    ):
        super().__init__()


        
        
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
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


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=4,
        use_batchnorm=True,
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
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
    
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
    
class Unet(SegmentationModel):
     def __init__(
        self,
        encoder_depth: int = 4,
        decoder_use_batchnorm: bool = True,
        decoder_channels: list[int] = (256, 128, 64, 32),
        classes: int = 2,
        activation = nn.Identity(),
        head_upsampling: int = 2,
    ):
        super().__init__()

        self.encoder = Encoder()

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            upsampling=head_upsampling,
        )
        self.name = f"u-Swin"
        self.initialize()