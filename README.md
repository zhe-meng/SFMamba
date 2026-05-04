# Spatial-Frequency Joint Learning Mamba for Hyperspectral Image Classification 2026 GRSL

PyTorch implementation of Spatial-Frequency Joint Learning Mamba for Hyperspectral Image Classification.

# Basic Usage

```
import torch
from SFMamba import SFMamba

# Take the Indian Pines dataset as an example, the number of
# classes and spectral channels are 16 and 200, respectively.
# Use common settings such as patch size = 11, batch size = 100,
# epochs = 100, and Adam (learning rate = 0.001) to achieve
# good results. In the paper, the default patch size is 12.

model = SFMamba(in_chans=200, num_classes=16)
model.eval()
print(model)
model = model.cuda()
input = torch.randn(100, 200, 11, 11)
input = input.cuda()
y = model(input)
print(y.size())
```

# Paper

[Spatial-Frequency Joint Learning Mamba for Hyperspectral Image Classification](https://ieeexplore.ieee.org/document/11343773)

If you find this code to be useful for your research, please consider citing.

```
@article{meng2026spatial,
  title={Spatial-frequency joint learning Mamba for hyperspectral image classification},
  author={Meng, Zhe and Yue, Lele and Zhao, Feng},
  journal={IEEE Geoscience and Remote Sensing Letters},
  volume={23},
  pages={1--5},
  year={2026},
  publisher={IEEE}
}
```


