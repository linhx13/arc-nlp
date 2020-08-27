import torch
from torch import nn
from torch.nn import functional as F

class CnnEncoder(nn.Module):

    def __init__(self,
                 input_dim, filters=100, kernel_sizes=(2, 3, 4, 5),
                 conv_layer_activation=None, output_dim=None):
        super(CnnEncoder, self).__init__()
        self.input_dim = input_dim
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=filters, kernel_size=ks)
            for ks in kernel_sizes])
        self.conv_layer_activation = conv_layer_activation or F.relu
        maxpool_output_dim = filters * len(kernel_sizes)
        if output_dim:
            self.projection_layer = nn.Linear(maxpool_output_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.projection_layer = None
            self.output_dim = maxpool_output_dim

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor=None):
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1)

        tokens = torch.transpose(tokens, 1, 2)
        filter_outputs = [self.conv_layer_activation(conv_layer(tokens)).max(dim=2)[0]
                          for conv_layer in self.conv_layers]
        maxpool_output = torch.cat(filter_outputs, dim=1) \
            if len(filter_outputs) > 1 else filter_outputs[0]
        if self.projection_layer:
            output = self.projection_layer(maxpool_output)
        else:
            output = maxpool_output
        return output
