import torch
from torch import nn
from torch.nn import functional as F


class CNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        num_filters=100,
        kernel_sizes=(1, 3, 5),
        conv_layer_activation=None,
        output_dim=None,
    ):
        super(CNNEncoder, self).__init__()
        for kernel_size in kernel_sizes:
            assert kernel_size % 2 == 1, "kernel size has to be odd numbers."
        self.input_dim = input_dim
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=num_filters,
                    kernel_size=ks,
                    padding=ks // 2,
                )
                for ks in kernel_sizes
            ]
        )
        self.conv_layer_activation = conv_layer_activation or F.relu
        maxpool_output_dim = num_filters * len(kernel_sizes)
        if output_dim:
            self.projection_layer = nn.Linear(maxpool_output_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.projection_layer = None
            self.output_dim = maxpool_output_dim

    def forward(
        self, inputs: torch.Tensor, mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        # [N, L, C] -> [N, C, L]
        inputs = torch.transpose(inputs, 1, 2)
        # [[N, C, L], ...]
        conv_outputs = [
            self.conv_layer_activation(conv_layer(inputs))
            for conv_layer in self.conv_layers
        ]
        if mask is not None:
            mask = mask.unsqueeze(1)  # [N, 1, L]
            conv_outputs = [
                x.masked_fill_(mask.eq(False), float("-inf"))
                for x in conv_outputs
            ]
        maxpool_outputs = [
            F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
            for x in conv_outputs
        ]  # [[N, C], ...]
        outputs = torch.cat(maxpool_outputs, dim=-1)  # [N, C]
        if self.projection_layer:
            outputs = self.projection_layer(outputs)
        return outputs
