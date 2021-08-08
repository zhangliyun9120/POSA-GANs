import torch
import torch.nn as nn
from models.bilinear import crop_bbox_batch, masking_feature
from torch.autograd import Variable
import torch.nn.functional as F
from graphviz import Digraph
from models.roi_layers import ROIAlign, ROIPool

import torch.backends.cudnn as cudnn


# show the graph of model and parameter size
def show_model_parameter(output, model, name=None):
    print("{} parameters: ".format(name))
    g = make_dot(output)
    g.view()

    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        print("lay：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("para：" + str(l))
        k = k + l
    print("total：" + str(k))


# 画pytorch模型图，以及参数计算
def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot


def get_z_random(batch_size, z_dim, random_type='gauss'):
    if random_type == 'uni':
        z = torch.rand(batch_size, z_dim) * 2.0 - 1.0
    elif random_type == 'gauss':
        z = torch.randn(batch_size, z_dim)
    return z


def transform_z_flat(batch_size, time_step, z_flat, obj_to_img):
    # restore z to batch with padding
    z = torch.zeros(batch_size, time_step, z_flat.size(1)).to(z_flat.device)
    for i in range(batch_size):
        idx = (obj_to_img.data == i).nonzero()
        if idx.dim() == 0:
            continue
        idx = idx.view(-1)
        n = idx.size(0)
        z[i, :n] = z_flat[idx]
    return z


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        # print("input_tensor.shape: ", input_tensor.shape)
        # print("h_cur.shape: ", h_cur.shape)
        # print("c_cur.shape: ", c_cur.shape)
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, device):
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device))


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        if isinstance(hidden_dim, list):
            num_layers = len(hidden_dim)
        elif isinstance(hidden_dim, int):
            num_layers = 1

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0), device=input_tensor.device)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class LayoutConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True, return_all_layers=False):
        super(LayoutConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        if isinstance(hidden_dim, list) or isinstance(hidden_dim, tuple):
            num_layers = len(hidden_dim)
        elif isinstance(hidden_dim, int):
            num_layers = 1

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size, input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        # cell_list is the number of objects
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, obj_tensor, obj_to_img, hidden_state=None):
        """

        Parameters
        ----------
        obj_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        # split input_tensor into list according to obj_to_img
        O = obj_tensor.size(0)
        previous_img_id = 0

        layouts_list = []
        temp = []
        for i in range(O):
            current_img_id = obj_to_img[i]
            if current_img_id == previous_img_id:
                temp.append(obj_tensor[i])
            else:
                temp = torch.stack(temp, dim=0)
                temp = torch.unsqueeze(temp, 0)
                layouts_list.append(temp)
                temp = [obj_tensor[i]]
                previous_img_id = current_img_id
        # append last one
        temp = torch.stack(temp, dim=0)
        temp = torch.unsqueeze(temp, 0)
        layouts_list.append(temp)

        N = len(layouts_list)
        all_layer_output_list, all_last_state_list = [], []
        for i in range(N):
            obj_tensor = layouts_list[i]
            hidden_state = self._init_hidden(batch_size=obj_tensor.size(0), device=obj_tensor.device)

            layer_output_list = []
            last_state_list = []

            seq_len = obj_tensor.size(1)
            cur_layer_input = obj_tensor

            for layer_idx in range(self.num_layers):

                h, c = hidden_state[layer_idx]
                output_inner = []
                for t in range(seq_len):
                    h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                     cur_state=[h, c])
                    output_inner.append(h)

                layer_output = torch.stack(output_inner, dim=1)
                cur_layer_input = layer_output

                layer_output_list.append(layer_output)
                last_state_list.append([h, c])

            if not self.return_all_layers:
                layer_output_list = layer_output_list[-1:]
                last_state_list = last_state_list[-1:]

            all_layer_output_list.append(layer_output_list)
            all_last_state_list.append(last_state_list)

        # concate last output to form a tensor
        batch_output = []
        for i in range(N):
            batch_output.append(all_last_state_list[i][0][0])
        batch_output = torch.cat(batch_output, dim=0)

        return batch_output

    def _init_hidden(self, batch_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


# style: (downsampling + global pooling + fc) --- (32,1,1)
class CropEncoder(nn.Module):
    def __init__(self, conv_dim=64, z_dim=256, class_num=133):
        # default: (cls, 3, 32, 32) -> (cls, 256, 1, 1)
        super(CropEncoder, self).__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # (3, 32, 32) -> (64, 32, 32)
        self.c1 = nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(conv_dim) if class_num == 0 else ConditionalBatchNorm2d(conv_dim, class_num)
        # (64, 32, 32) -> (128, 16, 16)
        self.c2 = nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(conv_dim * 2) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 2, class_num)
        # (128, 16, 16) -> (256, 8, 8)
        self.c3 = nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(conv_dim * 4) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 4, class_num)
        # (256, 8, 8) -> (512, 4, 4)
        self.c4 = nn.Conv2d(conv_dim * 4, conv_dim * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(conv_dim * 8) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 8, class_num)
        # # (512, 4, 4) -> (1024, 2, 2)
        self.conv5 = nn.Conv2d(conv_dim * 8, conv_dim * 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(conv_dim * 16) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 16, class_num)
        # pool (1024, 2, 2) -> (1024, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # (1024, 1, 1) -> (256, 1, 1)
        self.fc_mu = nn.Linear(conv_dim * 16, z_dim)
        self.fc_logvar = nn.Linear(conv_dim * 16, z_dim)

    def forward(self, imgs, objs=None):
        x = imgs
        x = self.c1(x)
        x = self.bn1(x) if objs is None else self.bn1(x, objs)
        x = self.activation(x)
        x = self.c2(x)
        x = self.bn2(x) if objs is None else self.bn2(x, objs)
        x = self.activation(x)
        x = self.c3(x)
        x = self.bn3(x) if objs is None else self.bn3(x, objs)
        x = self.activation(x)
        x = self.c4(x)
        x = self.bn4(x) if objs is None else self.bn4(x, objs)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.bn5(x) if objs is None else self.bn5(x, objs)
        x = self.activation(x)
        x = self.pool(x)

        # resize tensor to a flatten for fully convolution
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        std = logvar.mul(0.5).exp()
        eps = get_z_random(std.size(0), std.size(1)).to(imgs.device)
        z = eps.mul(std).add(mu)

        return z, mu, logvar


# style: (downsampling + global pooling + fc) --- (256,1,1)
class CropEncoder_s(nn.Module):
    def __init__(self, z_dim=256):
        super(CropEncoder_s, self).__init__()

        input_dim = 3
        dim = 64  # number of filters in the bottommost layer: 64
        n_downsample = 2  # number of downsampling layers in content encoder: 2
        n_res = 4  # number of residual blocks in content encoder/decoder: 4
        activ = 'lrelu'  # activation function [relu/lrelu/prelu/selu/tanh]: relu
        pad_type = 'reflect'  # padding type [zero/reflect]: reflect
        norm = 'in'

        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        self.model += [nn.Conv2d(dim, z_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


# content: (downsampling + residual blocks) --- (256, 8, 8)
class CropEncoder_c(nn.Module):
    def __init__(self, c_dim=512):
        super(CropEncoder_c, self).__init__()

        input_dim = 3
        dim = c_dim // 4  # number of filters in the bottommost layer: 64
        n_downsample = 2  # number of downsampling layers in content encoder: 2
        n_res = 4  # number of residual blocks in content encoder/decoder: 4
        activ = 'lrelu'  # activation function [relu/lrelu/prelu/selu/tanh]: relu
        pad_type = 'reflect'  # padding type [zero/reflect]: reflect
        norm = 'in'

        self.model = []
        # (3, 32, 32) -> (128, 32, 32)
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks (128, 32, 32) -> (512, 8, 8)
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks (512, 8, 8) -> (512, 8, 8)
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = c_dim  # 512

    def forward(self, x):
        return self.model(x)


class LayoutEncoder(nn.Module):
    def __init__(self, conv_dim=32, z_dim=256, embedding_dim=256, class_num=133, resi_num=6, clstm_layers=3):
        super(LayoutEncoder, self).__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        # self.embedding = nn.Embedding(class_num, embedding_dim)

        # hyperparameters setup:
        if clstm_layers == 1:
            self.clstm = LayoutConvLSTM(input_size=16, input_dim=512, hidden_dim=[64], kernel_size=(5, 5))
        elif clstm_layers == 2:
            self.clstm = LayoutConvLSTM(input_size=16, input_dim=512, hidden_dim=[128, 64], kernel_size=(5, 5))
        elif clstm_layers == 3:
            self.clstm = LayoutConvLSTM(input_size=16, input_dim=512, hidden_dim=[128, 64, 64], kernel_size=(5, 5))

        layers = []
        # Bottleneck layers: 6 layers res-blocks and keep dimension same.
        for i in range(resi_num):
            layers.append(ResidualBlock(dim_in=64, dim_out=64))
        self.residual = nn.Sequential(*layers)

        # (512, 256, 256) -> (32, 256, 256)
        self.c1 = nn.Conv2d(embedding_dim + z_dim, conv_dim, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv_dim) if class_num == 0 else ConditionalBatchNorm2d(conv_dim, class_num)
        # (32, 256, 256) -> (64, 128, 128)
        self.c2 = nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(conv_dim * 2) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 2, class_num)
        # (64, 128, 128) -> (128, 64, 64)
        self.c3 = nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(conv_dim * 4) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 4, class_num)
        # (128, 64, 64) -> (256, 32, 32)
        self.c4 = nn.Conv2d(conv_dim * 4, conv_dim * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(conv_dim * 8) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 8, class_num)

        # (256, 32, 32) -> (512, 16, 16)
        self.c5 = nn.Conv2d(conv_dim * 8, conv_dim * 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(conv_dim * 16) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 16, class_num)

    def forward(self, objs, obj_to_img, h):
        # embeddings = self.embedding(objs)
        # # print("embeddings.shape: ", embeddings.shape)  # torch.Size([7, 256])
        # fm_obj = fm_obj.squeeze(-1).squeeze(-1)   # (cls, 256, 1, 1) -> (cls, 256)
        # embeddings_fm_obj = torch.cat((embeddings, fm_obj), dim=1)
        # # print("embeddings_fm_obj.shape: ", embeddings_fm_obj.shape)   # torch.Size([cls, 512])
        #
        # print("fm_obj.shape: ", fm_obj.shape)  # torch.Size([cls, 512])
        # masks = masks.unsqueeze(1)
        # # print("masks.shape: ", masks.shape)   # torch.Size([cls, 1, 256, 256])
        # embeddings_fm_obj = embeddings_fm_obj.view(embeddings_fm_obj.size(0), embeddings_fm_obj.size(1), 1, 1)
        # # if hasattr(torch.cuda, 'empty_cache'):
        # #     torch.cuda.empty_cache()
        # h = embeddings_fm_obj * masks
        # # print("h.shape: ", h.shape)     # torch.Size([cls, 512, 256, 256])

        # downsample layout
        h = self.c1(h)
        h = self.bn1(h, objs)
        h = self.activation(h)
        h = self.c2(h)
        h = self.bn2(h, objs)
        h = self.activation(h)
        h = self.c3(h)
        h = self.bn3(h, objs)
        h = self.activation(h)
        h = self.c4(h)
        h = self.bn4(h, objs)

        h = self.c5(h)
        h = self.bn5(h, objs)
        # print("h.shape: ", h.shape)     # torch.Size([cls, 512, 16, 16])

        # clstm fusion (cls, 512, 16, 16) -> (cls, 64, 16, 16)
        h = self.clstm(h, obj_to_img)
        # residual block
        h = self.residual(h)

        return h


class Decoder(nn.Module):
    def __init__(self, conv_dim=32):
        super(Decoder, self).__init__()

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.activation_final = nn.Tanh()

        # (64, 16, 16) -> (512, 16, 16)
        self.c0 = nn.Conv2d(conv_dim * 2, conv_dim * 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(conv_dim * 16)

        # Upsampling:
        # (512, 16, 16) -> (256, 32, 32)
        self.dc1 = nn.ConvTranspose2d(conv_dim * 16, conv_dim * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv_dim * 8)
        # (256, 32, 32) -> (128, 64, 64)
        self.dc2 = nn.ConvTranspose2d(conv_dim * 8, conv_dim * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(conv_dim * 4)
        # (128, 64, 64) -> (64, 128, 128)
        self.dc3 = nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(conv_dim * 2)
        # (64, 128, 128) -> (32, 256, 256)
        self.dc4 = nn.ConvTranspose2d(conv_dim * 2, conv_dim * 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(conv_dim * 1)

        # (32, 256, 256) -> (3, 256, 256)
        self.c5 = nn.Conv2d(conv_dim * 1, 3, kernel_size=7, stride=1, padding=3, bias=True)

    def forward(self, hidden):
        # print("Decoder.forward: ")
        # print("hidden.shape: ", hidden.shape)   # torch.Size([1, 64, 16, 16])
        h = hidden
        h = self.c0(h)
        h = self.bn0(h)
        h = self.activation(h)
        h = self.dc1(h)
        h = self.bn1(h)
        h = self.activation(h)
        h = self.dc2(h)
        h = self.bn2(h)
        h = self.activation(h)
        h = self.dc3(h)
        h = self.bn3(h)
        h = self.activation(h)
        h = self.dc4(h)
        h = self.bn4(h)
        h = self.activation(h)
        h = self.c5(h)
        # trick: make generated image in range (-1,1)
        h = self.activation_final(h)
        return h


class ResnetRoI(nn.Module):
    def __init__(self, num_classes=133, input_dim=3, ch=64):
        super(ResnetRoI, self).__init__()
        self.num_classes = num_classes

        self.block1 = BasicBlock(3, ch, downsample=True)
        self.block2 = ResBlock_Downsample(ch, ch * 2, downsample=True)
        self.block3 = ResBlock_Downsample(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock_Downsample(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock_Downsample(ch * 8, ch * 8, downsample=True)
        self.block6 = ResBlock_Downsample(ch * 8, ch * 16, downsample=True)
        self.block7 = ResBlock_Downsample(ch * 16, ch * 16, downsample=False)
        self.l8 = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.activation = nn.ReLU()

        self.roi_align_s = ROIAlign((8, 8), 1.0 / 8.0, int(0))
        self.roi_align_l = ROIAlign((8, 8), 1.0 / 16.0, int(0))

        self.block_obj4 = ResBlock(ch * 4, ch * 8, downsample=False)
        self.block_obj5 = ResBlock(ch * 8, ch * 8, downsample=False)
        self.block_obj6 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 16))

    def forward(self, x, y=None, bbox=None):
        b = bbox.size(0)
        # 256x256
        x = self.block1(x)
        # 128x128
        x = self.block2(x)
        # 64x64
        x1 = self.block3(x)
        # 32x32
        x2 = self.block4(x1)
        # 16x16
        x = self.block5(x2)
        # 8x8
        x = self.block6(x)
        # 4x4
        x = self.block7(x)
        x = self.activation(x)
        x = torch.sum(x, dim=(2, 3))
        out_im = self.l8(x)

        # obj path
        # seperate different path
        s_idx = ((bbox[:, 3] - bbox[:, 1]) < 128) * ((bbox[:, 4] - bbox[:, 2]) < 128)
        bbox_l, bbox_s = bbox[~s_idx], bbox[s_idx]
        y_l, y_s = y[~s_idx], y[s_idx]

        obj_feat_s = self.block_obj4(x1)
        obj_feat_s = self.block_obj5(obj_feat_s)
        obj_feat_s = self.roi_align_s(obj_feat_s, bbox_s)

        obj_feat_l = self.block_obj5(x2)
        obj_feat_l = self.roi_align_l(obj_feat_l, bbox_l)

        obj_feat = torch.cat([obj_feat_l, obj_feat_s], dim=0)
        y = torch.cat([y_l, y_s], dim=0)
        obj_feat = self.block_obj6(obj_feat)
        obj_feat = self.activation(obj_feat)
        obj_feat = torch.sum(obj_feat, dim=(2, 3))
        out_obj = self.l_obj(obj_feat)
        out_obj = out_obj + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)

        return out_im, out_obj


class Generator(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=256, z_dim=256, c_dim=512, obj_size=32, clstm_layers=3):
        super(Generator, self).__init__()

        # print("Generator Model:")

        n_res = 4  # number of residual blocks in content encoder/decoder: 4
        activ = 'relu'  # activation function [relu/lrelu/prelu/selu/tanh]: relu
        pad_type = 'reflect'  # padding type [zero/reflect]: reflect
        mlp_dim = 512  # number of filters in MLP: 512

        self.backbone = ResnetRoI(num_classes=num_embeddings, input_dim=3)

        self.obj_size = obj_size

        # Object Estimator for style:
        self.crop_encoder = CropEncoder(z_dim=z_dim, class_num=num_embeddings)
        # print("style_encoder model:", self.crop_encoder)

        # Object Estimator for style:
        # self.crop_encoder_s = CropEncoder_s(z_dim=z_dim)
        # print("Style_encoder model:", self.crop_encoder_s)

        # Object Estimator for content:
        self.crop_encoder_c = CropEncoder_c(c_dim=c_dim)
        # print("Content_encoder model:", self.crop_encoder_c)

        # word embedding
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        # mapping function
        mapping = list()
        self.mapping = nn.Sequential(*mapping)

        # Combine style and content
        self.combinb = Com_S_C(n_res, self.crop_encoder_c.output_dim, res_norm='adain', activ=activ, pad_type=pad_type)
        # print("combinb fm model:", self.combinb)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(z_dim + embedding_dim, self.get_num_adain_params(self.combinb), mlp_dim, 3, norm='none', activ=activ)
        # print("MLP model:", self.mlp)

        # Objects Fuser:
        self.layout_encoder = LayoutEncoder(z_dim=z_dim, embedding_dim=embedding_dim, class_num=num_embeddings, clstm_layers=clstm_layers)
        # print("layout model:", self.layout_encoder)

        # Decoder:
        self.decoder = Decoder()
        # self.apply(weights_init)
        # print("Decoder model:", self.decoder)

    def forward(self, img, objs, boxes, masks, obj_to_img):
        # 1, Object Style and Content Encoder:
        _, c_en, s_en, _, _ = self.encode(img, objs, boxes, obj_to_img)

        # 2, decode object-feature-maps combined from content and style codes to an whole image
        img_rec = self.decode(c_en, s_en, objs, boxes, masks, obj_to_img)

        return img_rec

    def crop(self, imgs, boxes, obj_to_img):
        # print("In crop: ")
        # print("imgs.shape: ", imgs.shape)  # torch.Size([1, 3, 256, 256])
        # print("boxes.shape: ", boxes.shape)  # torch.Size([cls, 4])
        # print("obj_to_img.shape: ", obj_to_img.shape)  # torch.Size([cls])
        # print("z_rand.shape: ", z_rand.shape)  # torch.Size([cls, 32])
        # 0, crops:
        crops_result = crop_bbox_batch(imgs, boxes, obj_to_img, self.obj_size)
        # print("crops_result.shape: ", crops_result.shape)  # torch.Size([cls, 3, 32, 32])
        return crops_result

    def encode(self, img, objs, boxes, obj_to_img):
        # 0, crops:
        crops_input = self.crop(img, boxes, obj_to_img)
        # print("In encoder: ")
        # print("objs.shape: ", objs.shape)  # torch.Size([cls])
        # 1, Object Style Estimator:
        style, mu, logvar = self.crop_encoder(crops_input, objs)
        # print("style.shape: ", style.shape)  # torch.Size([cls, 256])
        # print("mu.shape: ", mu.shape)  # torch.Size([cls, 256])
        # print("logvar.shape: ", logvar.shape)  # torch.Size([cls, 256])
        # 2, Object Content feature extraction:
        content = self.crop_encoder_c(crops_input)
        # print("content.shape: ", content.shape)  # torch.Size([cls, 256, 8, 8])
        return crops_input, content, style, mu, logvar

    def combine(self, content, style, objs):
        # print("In combine: ")
        # concatenate style with word
        embeddings = self.embedding(objs)
        # print("embeddings.shape: ", embeddings.shape)  # torch.Size([cls, 256])
        style = style.squeeze(-1).squeeze(-1)  # (cls, 256, 1, 1) -> (cls, 256)
        embeddings_style = torch.cat((embeddings, style), dim=1)
        # print("embeddings_style.shape: ", embeddings_style.shape)   # torch.Size([cls, 512])
        w = self.mapping(embeddings_style)

        adain_params = self.mlp(w)
        # print("after mlp adain_params:", adain_params.size())  # torch.Size([cls, 4096])
        # show_model_parameter(adain_params, self.mlp, name='mlp')  # total 32:19008;  256：144448
        self.assign_adain_params(adain_params, self.combinb)
        fm_objs = self.combinb(content)
        # print("fm.shape: ", fm.shape)  # torch.Size([cls, 256, 1, 1])
        # show_model_parameter(fm, self.combinb, name='combinb')  # total 32：107872;  256：6884096
        return fm_objs

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

    def decode(self, c_en, s_en, objs, boxes, masks, obj_to_img):
        # print("In decoder: ")
        # 0, combine object content and style to a feature map
        # s_en = s_en.unsqueeze(-1).unsqueeze(-2)
        fm_objs = self.combine(c_en, s_en, objs)
        # print("fm_objs.shape: ", fm_objs.shape)  # torch.Size([cls, 256, 8, 8])

        # feature masking:
        # print("masks.shape: ", masks.shape)  # torch.Size([cls, 256, 256])
        masked_feature = masking_feature(boxes, fm_objs, masks)

        # 1, Objects Fuser: assemble all-objects-fm to a whole-image-fm
        fm_fusion = self.layout_encoder(objs, masks, obj_to_img, masked_feature)
        # print("fm_fusion.shape: ", fm_fusion.shape)  # torch.Size([1, 64, 16, 16])
        # show_model_parameter(fm_fusion, self.layout_encoder, name='layout_encoder')  # total：4901696
        # 2, Decoder: from fm to decoder a image
        image_fake = self.decoder(fm_fusion)
        # print("image_fake.shape: ", image_fake.shape)  # torch.Size([1, 3, 256, 256])
        # show_model_parameter(image_fake, self.decoder, name='decoder')  # total：927811
        return image_fake


##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='lrelu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Com_S_C(nn.Module):
    def __init__(self, n_res, dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Com_S_C, self).__init__()

        # AdaIN residual blocks
        self.res1 = ResBlock_upsample(dim, dim // 2, upsample=True)
        self.res2 = ResBlock_upsample(dim // 2, dim // 4, upsample=True)

        # self.model = []
        # self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # self.model = nn.Sequential(*self.model)

    def forward(self, x):
        # 16x16
        x = self.res1(x)
        # 32x32
        x = self.res2(x)
        return x
        # return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


##################################################################################
# Basic Blocks
##################################################################################
class ResBlock_upsample(nn.Module):
    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1, upsample=False):
        super(ResBlock_upsample, self).__init__()
        self.upsample = upsample
        self.h_ch = h_ch if h_ch else out_ch
        self.conv1 = conv2d(in_ch, self.h_ch, ksize, pad=pad)
        self.conv2 = conv2d(self.h_ch, out_ch, ksize, pad=pad)
        self.b1 = AdaptiveInstanceNorm2d(in_ch)
        self.b2 = AdaptiveInstanceNorm2d(self.h_ch)
        self.learnable_sc = in_ch != out_ch or upsample
        if self.learnable_sc:
            self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)
        self.activation = nn.ReLU()

    def residual(self, in_feat):
        x = in_feat
        x = self.b1(x)
        x = self.activation(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.b2(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x

    def shortcut(self, x):
        if self.learnable_sc:
            if self.upsample:
                x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.c_sc(x)
        return x

    def forward(self, in_feat):
        return self.residual(in_feat) + self.shortcut(in_feat)


class ResBlock_Downsample(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
        super(ResBlock_Downsample, self).__init__()
        self.conv1 = conv2d(in_ch, out_ch, ksize, 1, pad)
        self.conv2 = conv2d(out_ch, out_ch, ksize, 1, pad)
        self.activation = nn.ReLU()
        self.downsample = downsample
        self.learnable_sc = (in_ch != out_ch) or downsample
        if self.learnable_sc:
            self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)

    def residual(self, in_feat):
        x = in_feat
        x = self.conv1(self.activation(x))
        x = self.conv2(self.activation(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
        return x

    def forward(self, in_feat):
        return self.residual(in_feat) + self.shortcut(in_feat)


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv2d(in_ch, out_ch, ksize, 1, pad)
        self.conv2 = conv2d(out_ch, out_ch, ksize, 1, pad)
        self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)
        self.activation = nn.ReLU()
        self.downsample = downsample

    def forward(self, in_feat):
        x = in_feat
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x + self.shortcut(in_feat)

    def shortcut(self, x):
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return self.c_sc(x)


def conv2d(in_feat, out_feat, kernel_size=3, stride=1, pad=1, spectral_norm=True):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='lrelu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='lrelu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='lrelu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


##################################################################################
# VGG network definition
##################################################################################
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        # relu4_3 = h

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        return relu5_3
        # return [relu1_2, relu2_2, relu3_3, relu4_3]


##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


if __name__ == '__main__':
    # from torchsummary import summary
    # from torchsummaryX import summary

    from data.t2v_custom_mask import get_dataloader

    cudnn.enabled = True
    cudnn.benchmark = True

    # 'cuda:0' for multi-gpu 0, only one gpu just 'cuda'
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    obj_size = 32
    embedding_dim = 256
    z_dim = 256
    c_dim = 256
    batch_size = 1
    clstm_layers = 1

    n_res = 4  # number of residual blocks in content encoder/decoder: 4
    activ = 'lrelu'  # activation function [relu/lrelu/prelu/selu/tanh]: relu
    pad_type = 'reflect'  # padding type [zero/reflect]: reflect
    mlp_dim = 256  # number of filters in MLP: 32

    # 创建训练以及测试得数据迭代器
    loader_t, loader_v = get_dataloader(batch_size=1, T2V_DIR='../datasets/thermal2visible_256x256', is_training=True)

    # print("0 memory allocated in MB:{}".format(torch.cuda.memory_allocated() / 1024 ** 2))  # 0.0

    # using coco panoptic categories.json
    vocab_num = 133

    model_CropEncoder = CropEncoder(z_dim=z_dim, class_num=vocab_num).cuda()

    # model_CropEncoder_c = CropEncoder_c(c_dim=c_dim).cuda()
    #
    # model_layout_encoder = LayoutEncoder(z_dim=z_dim, embedding_dim=embedding_dim, class_num=vocab_num,
    #                                      clstm_layers=clstm_layers).cuda()
    #
    # model_decoder = Decoder().cuda()

    # 生成网络模型
    # netG_t = Generator(num_embeddings=vocab_num, embedding_dim=embedding_dim, z_dim=z_dim, c_dim=c_dim,
    #                    clstm_layers=clstm_layers).cuda()
    netG_v = Generator(num_embeddings=vocab_num, embedding_dim=embedding_dim, z_dim=z_dim, c_dim=c_dim,
                       clstm_layers=clstm_layers).cuda()

    for i, (batch_t, batch_v) in enumerate(zip(loader_t, loader_v)):
        img_t, objs, boxes, masks, obj_to_img = batch_t
        img_v, _, _, _, _ = batch_v
        style_rand_t = torch.randn(objs.size(0), z_dim).cuda()
        style_rand_v = torch.randn(objs.size(0), z_dim).cuda()
        img_t, img_v, objs, boxes, masks, obj_to_img = img_t.cuda(), img_v.cuda(), objs.cuda(), boxes.cuda(), \
                                                       masks.cuda(), obj_to_img.cuda()

        # print("1 memory allocated in MB:{}".format(torch.cuda.memory_allocated() / 1024 ** 2))  # 4.21435546875

        # test Crop
        crops_result = crop_bbox_batch(img_v, boxes, obj_to_img, obj_size)  # ([cls, 3, 32, 32])

        # test CropEncoder style
        print("crops_result.shape:{}, objs.shape:{}".format(crops_result.shape, objs.shape))
        style, mu, logvar = model_CropEncoder(crops_result, objs)
        print("style.shape:{}, mu.shape:{}, logvar.shape:{}".format(style.shape, mu.shape, logvar.shape))  # ([cls, 256])
        # summary(model_CropEncoder, crops_result, objs)
        show_model_parameter(style, model_CropEncoder, name='style_encoder')
        # total 64-128d:3195280; 32-256d:12.2031M=12203072

        # # test CropEncoder content
        # content = model_CropEncoder_c(crops_result)
        # # print("content.shape: ", content.shape)  # torch.Size([cls, 256, 8, 8])
        # summary(model_CropEncoder_c, input_size=(3, 32, 32))    # torchsummary
        # # summary(model_CropEncoder_c, crops_result)    # torchsummaryX
        # show_model_parameter(content, model_CropEncoder_c, name='content_encoder')
        # # total 64-128d:1355296; 32-256d:5.38586M=5385856

        # test adain_params = self.mlp(style)
        # print("after mlp adain_params:", adain_params.size())  # torch.Size([cls, 4096])
        # show_model_parameter(adain_params, self.mlp, name='mlp')  # total 32:19008;  256：144448

        # test assign_adain_params(adain_params, self.combinb)
        # fm = self.combinb(content)
        # print("fm.shape: ", fm.shape)  # torch.Size([cls, 256, 1, 1])
        # show_model_parameter(fm, self.combinb, name='combinb')  # total 32：107872;  256：6884096

        # # test LayoutEncoder
        # fm_objs = torch.randn(objs.size(0), 256, 1, 1).cuda()
        # fm_fusion = model_layout_encoder(objs, masks, obj_to_img, fm_objs)
        # print("fm_fusion.shape: ", fm_fusion.shape)  # torch.Size([1, 64, 16, 16])
        # show_model_parameter(fm_fusion, model_layout_encoder, name='layout_encoder')
        # # total 64-128d-3layers:54436864; 32-256d-3layers:13784512

        # # test Decoder
        # fm_fusion = torch.randn(1, 64, 16, 16).cuda()
        # image_fake = model_decoder(fm_fusion)
        # # print("image_fake.shape: ", image_fake.shape)  # torch.Size([1, 3, 256, 256])
        # summary(model_decoder, fm_fusion)  # torchsummaryX
        # # summary(model_decoder, input_size=(64, 16, 16))    # torchsummary
        # show_model_parameter(image_fake, model_decoder, name='decoder')
        # # total 64-128d：12334147; 32-256d:3086883

        # test Generator
        output_v = netG_v(img_v, objs, boxes, masks, obj_to_img)
        # print("output_v.shape: ", output_v.shape)  # torch.Size([1, 3, 256, 256])
        # show_model_parameter(output_v, netG_v, name='generator')
        # total 64-128d:73668467; 32-256d:42528675
