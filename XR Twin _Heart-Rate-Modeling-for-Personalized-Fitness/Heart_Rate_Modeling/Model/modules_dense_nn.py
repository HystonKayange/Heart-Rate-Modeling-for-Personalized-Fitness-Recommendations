import torch
import torch.nn as nn

class DenseNN(nn.Module):
    def __init__(self, *dim_layers, activation=nn.ReLU(), output_activation=None, dim_context=0, bias=True, output_bounds=None):
        super(DenseNN, self).__init__()
        self.dim_layers = list(dim_layers)
        self.dim_context = dim_context

        if dim_context:
            self.dim_layers[0] += dim_context
        layers = []
        for l, r in zip(self.dim_layers[:-1], self.dim_layers[1:]):
            layers.append(nn.Linear(l, r, bias))
            layers.append(activation)
        self.layers = nn.ModuleList(layers)
        self.output_activation = output_activation if output_activation else nn.Identity()
        self.output_bounds = output_bounds

    def forward(self, x, context=None):
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        for layer in self.layers:
            x = layer(x)
        x = self.output_activation(x)
        if self.output_bounds is not None:
            x = torch.sigmoid(x) * (self.output_bounds[1] - self.output_bounds[0]) + self.output_bounds[0]
        return x

class PersonalizedScalarNN(DenseNN):
    def __init__(self, *dim_layers, personalization='none', dim_personalization=0, activation=nn.ReLU(), output_activation=None):
        super().__init__(*dim_layers, activation=activation, output_activation=output_activation)
        self.personalization = personalization
        self.dim_personalization = dim_personalization if personalization != 'none' else 0

    def forward(self, x, context=None):
        if self.personalization == 'concatenate' and context is not None:
            x = torch.cat([x, context], dim=-1)
        return super().forward(x)
