from torch import nn


class ConvBranch(nn.Module):

    def __init__(self, in_filters, out_filters, kernel, separable):
        super(ConvBranch, self).__init__()
        self.separable = separable
        self.in_fitlers = in_filters
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
                nn.Conv2d(self.in_filters, self.in_filters, kernel_size=kernel, padding=self.padding,
                          groups=self.in_fitlers, bias=False),
                # depthwise
                nn.Conv2d(self.in_fitlers, self.out_filters, kernel_size=1, bias=False),  # pointwise
                nn.BatchNorm2d(out_filters, track_running_stats=False),
                nn.ReLU()
            )

        else:

            self.conv2 = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_filters, track_running_stats=False),

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


class fixedEnasLayer(nn.Module):

    def __init__(self, in_filters, out_filters, layer_id, prev_layers, layer_type):
        super(fixedEnasLayer, self).__init__()
        self.in_filers = in_filters
        self.out_filters = out_filters
        self.layer_id = layer_id
        self.prev_layers = prev_layers
        self.layer_type = layer_type

        if layer_type == 1:
            self.layer = ConvBranch(self.in_filers, self.out_filters, 3, False)
        elif layer_type == 2:
            self.layer = ConvBranch(self.in_filers, self.out_filters, 5, False)
        elif layer_type == 3:
            self.layer = ConvBranch(self.in_filers, self.out_filters, 3, True)
        elif layer_type == 4:
            self.layer = ConvBranch(self.in_filers, self.out_filters, 5, True)
        elif layer_type == 5:
            self.layer = PoolBranch(self.in_filers, self.out_filters, 3, "max")
        elif layer_type == 6:
            self.layer = ConvBranch(self.in_filers, self.out_filters, 3, "avg")

    def forward(self, x):
        return self.layer(x)


class SharedEnasLayer(nn.Module):

    def __init__(self, in_filters, out_filters, layer_id, prev_layers, layer_type):
        super(fixedEnasLayer, self).__init__()
        self.in_filers = in_filters
        self.out_filters = out_filters
        self.layer_id = layer_id
        self.prev_layers = prev_layers
        self.layer_type = layer_type

        raise NotImplemented(f"SharedEnasLayer:init")

    def forward(self, x):
        raise NotImplemented(f"SharedEnasLayer:forward")
