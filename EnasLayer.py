from torch import nn
import torch
import torch.nn.functional as F


class FactorizedReduction(nn.Module):
    def __init__(self, in_filters, out_filters, stride, is_training):
        super(FactorizedReduction).__init__()

        assert out_filters % 2 == 0, ("FactorizedReduction: not even number of out_filters")

        self.out_filters = out_filters
        self.in_filters = in_filters
        self.stride = stride

        if self.stride == 1:
            self.path_conv = nn.Conv2d(in_filters, out_filters, kernel_size=1, bias=False)

        self.path1_pool = nn.AvgPool2d(kernel_size=1, stride=self.stride)
        self.path1_conv = nn.Conv2d(in_filters, out_filters // 2, kernel_size=1, bias=False)

        self.path2_pool = nn.AvgPool2d(kernel_size=1, stride=self.stride)
        self.path2_conv = nn.Conv2d(in_filters, out_filters // 2, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(out_filters, track_running_stats=False)

    def forward(self, x):
        if self.stride == 1:
            out = self.path_conv(x)
            out = self.bn(out)
        else:
            x_1 = self.path1_pool(x)
            x_1 = self.path1_conv(x_1)

            x_2 = F.pad(x, pad=[0, 1, 0, 1], mode='constant', value=0.)
            x_2 = x_2[:, :, 1:, 1:]
            x_2 = self.path2_pool(x_2)
            x_2 = self.path2_conv(x_2)

            out = torch.cat([x_1, x_2])
            out = self.bn(out)

        return out

    # def _factorized_reduction(self, x, out_filters, stride, is_training):
    #     """Reduces the shape of x without information loss due to striding."""
    #     assert out_filters % 2 == 0, (
    #         "Need even number of filters when using this factorized reduction.")
    #     if stride == 1:
    #         with tf.variable_scope("path_conv"):
    #             inp_c = self._get_C(x)
    #             w = create_weight("w", [1, 1, inp_c, out_filters])
    #             x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
    #                              data_format=self.data_format)
    #             x = batch_norm(x, is_training, data_format=self.data_format)
    #             return x


class ConvBranch(nn.Module):

    def __init__(self, in_filters, out_filters, kernel, separable):
        super(ConvBranch, self).__init__()
        self.separable = separable
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.kernel = kernel
        self.padding = (self.kernel - 1) // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_filters, track_running_stats=False),
            nn.ReLU()
        )

        if self.separable:

            self.conv2 = nn.Sequential(
                nn.Conv2d(self.in_filters, self.in_filters, kernel_size=self.kernel, padding=self.padding,
                          groups=self.in_filters, bias=False),
                # depthwise
                nn.Conv2d(self.in_filters, self.out_filters, kernel_size=1, bias=False),  # pointwise
                nn.BatchNorm2d(self.out_filters, track_running_stats=False),
                nn.ReLU()
            )

        else:

            self.conv2 = nn.Sequential(
                nn.Conv2d(self.in_filters, self.out_filters, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_filters, track_running_stats=False),

                nn.ReLU()
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


class PoolBranch(nn.Module):

    def __init__(self, in_filters, out_filters, kernel, pool_type):
        super(PoolBranch, self).__init__()
        self.kernel = kernel
        self.pool_type = pool_type
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_filters, track_running_stats=False),
            nn.ReLU()
        )

        if pool_type == "avg":

            self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        elif pool_type == "max":

            self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)

        return out


class FixedEnasLayer(nn.Module):

    def __init__(self, in_filters, out_filters, layer_id, prev_layers, layer_type):
        super(FixedEnasLayer, self).__init__()
        self.in_filers = in_filters
        self.out_filters = out_filters
        self.layer_id = layer_id
        self.prev_layers = prev_layers
        self.layer_type = layer_type

        if layer_type == 0:
            self.layer = ConvBranch(self.in_filers, self.out_filters, 3, False)
        elif layer_type == 1:
            self.layer = ConvBranch(self.in_filers, self.out_filters, 5, False)
        elif layer_type == 2:
            self.layer = ConvBranch(self.in_filers, self.out_filters, 3, True)
        elif layer_type == 3:
            self.layer = ConvBranch(self.in_filers, self.out_filters, 5, True)
        elif layer_type == 4:
            self.layer = PoolBranch(self.in_filers, self.out_filters, 3, "max")
        elif layer_type == 5:
            self.layer = PoolBranch(self.in_filers, self.out_filters, 3, "avg")
        else:
            raise AssertionError("layer_type must be in [0,5]")

    def forward(self, x):
        return self.layer(x)


class SharedEnasLayer(nn.Module):

    def __init__(self, in_filters, out_filters, layer_id, prev_layers, layer_type):
        super(SharedEnasLayer, self).__init__()
        self.in_filers = in_filters
        self.out_filters = out_filters
        self.layer_id = layer_id
        self.prev_layers = prev_layers
        self.layer_type = layer_type

        raise NotImplemented(f"SharedEnasLayer:init")

    def forward(self, x):
        raise NotImplemented(f"SharedEnasLayer:forward")
