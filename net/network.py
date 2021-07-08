from net.net_parts import *


class ISTANetPlus(nn.Module):
    def __init__(self, num_layers, rank):
        super(ISTANetPlus, self).__init__()
        self.num_layers = num_layers
        self.rank = rank

        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(BasicBlock(self.rank))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, under_img, mask):
        x = under_img
        layers_sym = []
        for i in range(self.num_layers):
            [x, layer_sym] = self.layers[i](x, under_img, mask)
            layers_sym.append(layer_sym)
        x_final = x
        return [x_final, layers_sym]


class ParallelNetwork(nn.Module):
    def __init__(self, num_layers, rank):
        super(ParallelNetwork, self).__init__()
        self.num_layers = num_layers
        self.rank = rank

        self.up_network = ISTANetPlus(self.num_layers, self.rank)
        self.down_network = ISTANetPlus(self.num_layers, self.rank)

    def forward(self, under_img_up, mask_up, under_img_down, mask_down):
        output_up, loss_layers_up = self.up_network(under_img_up, mask_up)
        output_down, loss_layers_down = self.down_network(under_img_down, mask_down)
        return output_up, loss_layers_up, output_down, loss_layers_down
