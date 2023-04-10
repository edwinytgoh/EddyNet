import torch

from eddynet import EddyNet

dependencies = ["torch"]

model_urls = {
    "eddynet": "https://raw.githubusercontent.com/edwinytgoh/eddynet/master/weights/eddynet_trained_on_aviso_ssh_1998-2018.pt",
}


def eddynet(pretrained=False, num_classes=3, num_filters=16, kernel_size=3):
    model = EddyNet(num_classes, num_filters, kernel_size)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls["eddynet"], map_location="cpu"
        )
        model.load_state_dict(state_dict)
    return model
