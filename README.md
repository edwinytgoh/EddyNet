# EddyNet

EddyNet is a deep neural network for pixel-wise classification of oceanic eddies, based on the paper "EddyNet: A Deep Neural Network For Pixel-Wise Classification of Oceanic Eddies" by R. Lguensat et al. The authors' original TensorFlow implementation is available at [https://github.com/redouanelg/EddyNet](https://github.com/redouanelg/EddyNet).

# Usage

To use EddyNet in your project, simply call `torch.hub.load("edwinytgoh/eddynet", "eddynet", pretrained=True, num_classes=3)`.

The `pretrain` argument is used to specify whether to load the pre-trained weights (default is `True`). The `num_classes` argument specifies the number of classes to predict (2 for binary classification, 3 for not eddy, cyclonic, or anticyclonic).

## Example

Here's an example of how to use EddyNet to generate a segmentation mask for eddies from satellite-derived absolute dynamic topography (ADT) data. An example NetCDF file from the [Copernicus daily sea level product](https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-sea-level-global?tab=overview) is provided in the [data](./data) directory under the name `dt_global_twosat_phy_l4_20220428_vDT2021.nc`.

```python
import torch
import xarray as xr

# Load EddyNet with pre-trained weights and send to GPU if available
model = torch.hub.load("edwinytgoh/eddynet", "eddynet", pretrained=True, num_classes=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Obtain filtered ADT data as a numpy array
ds = xr.open_dataset("dt_global_twosat_phy_l4_20220428_vDT2021.nc")
adt = ds["adt"].values.squeeze()

# Convert ADT array to PyTorch tensor with batch dimension and channel dimension, each of size 1
adt_tensor = torch.from_numpy(adt).reshape(1, 1, *adt.shape).float().to(device)

# Feed ADT tensor into EddyNet and obtain eddy mask
logits = model(adt_tensor)
eddy_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
```


