import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(63, 1000),
            torch.nn.BatchNorm1d(1000),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),

            torch.nn.Linear(1000, 500),
            torch.nn.BatchNorm1d(500),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),

            torch.nn.Linear(500, 200),
            torch.nn.BatchNorm1d(200),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),

            torch.nn.Linear(200, 50),
            torch.nn.ReLU(),

            torch.nn.Linear(50, 3)
        )

    def forward(self, x):
        return self.layers(x)


def test():
    model = Model()
    noise = torch.randn((20, 63))
    out = model(noise)
    print(out.shape)
