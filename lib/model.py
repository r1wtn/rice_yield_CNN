import torch
from torch import nn, rand
from collections import OrderedDict


def save_model(path, epoch_id, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch_id,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)


def load_state_dict(model, checkpoint_path, only_reg=False):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint["state_dict"]
    if only_reg:
        new_state_dict = OrderedDict()
        for layer_name in state_dict:
            layer_group_name = layer_name.split(".")[0].split("_")[0]
            if layer_group_name != "re":
                new_state_dict[layer_name] = state_dict[layer_name]
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=True)


class RiceYieldCNN(nn.Module):
    def __init__(self):
        super(RiceYieldCNN, self).__init__()

        ###definition of RiceYieldCNN###
        ###  block_0  ###
        self.conv_1 = nn.Conv2d(3, 45, (3, 3), stride=(1, 1), padding=(1, 1))
        self.pool_1 = nn.AvgPool2d((2, 1), stride=(2, 1))
        self.norm_1 = nn.BatchNorm2d(45)
        self.act_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(45, 25, (3, 3), stride=(1, 1), padding=(1, 1))
        self.norm_2 = nn.BatchNorm2d(25)
        self.act_2 = nn.LeakyReLU(0.1)
        self.pool_2 = nn.MaxPool2d((2, 2), stride=(2, 2))

        ###  block_1  ###
        self.conv_3 = nn.Conv2d(25, 50, (3, 3), stride=(1, 1), padding=(1, 1))
        self.norm_3 = nn.BatchNorm2d(50)
        self.pool_3 = nn.AvgPool2d((2, 3), stride=(2, 3))
        self.norm_4 = nn.BatchNorm2d(50)
        self.act_3 = nn.ReLU()
        self.pool_4 = nn.MaxPool2d((3, 3), stride=(3, 3))

        ###  block_2  ###
        self.conv_4 = nn.Conv2d(25, 25, (3, 3), stride=(1, 1), padding=(1, 1))
        self.norm_5 = nn.BatchNorm2d(25)
        self.pool_5 = nn.AvgPool2d((2, 3), stride=(2, 3))
        self.norm_6 = nn.BatchNorm2d(25)
        self.act_4 = nn.ReLU()
        self.pool_6 = nn.MaxPool2d((3, 3), stride=(3, 3))

        ###  block_3  ###
        self.conv_5 = nn.Conv2d(50, 16, (1, 1), stride=(1, 1), padding=(1, 1))
        self.norm_7 = nn.BatchNorm2d(16)
        self.act_5 = nn.ELU(1.0)

        ###  block_4  ###
        self.conv_6 = nn.Conv2d(75, 16, (1, 1), stride=(1, 1), padding=(1, 1))
        self.norm_8 = nn.BatchNorm2d(16)
        self.act_6 = nn.ELU(1.0)

        ###  block_5  ###
        self.conv_7 = nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=(1, 1))
        self.pool_7 = nn.AvgPool2d((2, 2), stride=(2, 2))
        self.norm_9 = nn.BatchNorm2d(16)
        self.act_7 = nn.ReLU()

        ###  block_6  ###
        self.conv_8 = nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=(1, 1))
        self.norm_10 = nn.BatchNorm2d(16)
        self.act_8 = nn.ReLU()
        self.conv_9 = nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=(1, 1))
        self.pool_8 = nn.AvgPool2d((2, 2), stride=(2, 2))
        self.norm_11 = nn.BatchNorm2d(16)

        ###  block_final  ###
        self.flat = nn.Flatten()
        self.fc = nn.Linear(2640, 1)
        self.act_9 = nn.ReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = self.norm_1(x)
        x = self.act_1(x)
        x = self.conv_2(x)
        x = self.norm_2(x)
        x = self.act_2(x)
        x = self.pool_2(x)

        x_1 = x.clone()
        x_1 = self.conv_3(x_1)
        x_1 = self.norm_3(x_1)
        x_1 = self.pool_3(x_1)
        x_1 = self.norm_4(x_1)
        x_1 = self.act_3(x_1)
        x_1 = self.pool_4(x_1)

        x_2 = x.clone()
        x_2 = self.conv_4(x_2)
        x_2 = self.norm_5(x_2)
        x_2 = self.pool_5(x_2)
        x_2 = self.norm_6(x_2)
        x_2 = self.act_4(x_2)
        x_2 = self.pool_6(x_2)

        x_3 = self.conv_5(x_1)
        x_3 = self.norm_7(x_3)
        x_3 = self.act_5(x_3)

        x_4 = torch.cat([x_1, x_2], dim=1)
        x_4 = self.conv_6(x_4)
        x_4 = self.norm_8(x_4)
        x_4 = self.act_6(x_4)

        x_5 = torch.mul(x_3, x_4)
        x_5 = self.conv_7(x_5)
        x_5 = self.pool_7(x_5)
        x_5 = self.norm_9(x_5)
        x_5 = self.act_7(x_5)

        x_6 = self.conv_8(x_4)
        x_6 = self.norm_10(x_6)
        x_6 = self.act_8(x_6)
        x_6 = self.conv_9(x_6)
        x_6 = self.pool_8(x_6)
        x_6 = self.norm_11(x_6)

        x_m = torch.add(x_5, x_6)
        x_m = self.flat(x_m)
        x_m = self.fc(x_m)
        out = self.act_9(x_m)

        return out


if __name__ == "__main__":
    model = RiceYieldCNN()
    model.eval()

    img = rand((1, 3, 512, 512), requires_grad=True)

    out = model(img)
    print(out)
