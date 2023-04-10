import torch

from eddynet import EddyNet

dependencies = ["torch"]


def eddynet(pretrained=False, num_classes=3, num_filters=16, kernel_size=3):
    model = EddyNet(num_classes, num_filters, kernel_size)
    if pretrained:
        model.load_state_dict(
            torch.load(
                "weights/eddynet_trained_on_aviso_ssh_1998-2018.pt",
                map_location=torch.device("cpu"),
            )
        )
    return model
