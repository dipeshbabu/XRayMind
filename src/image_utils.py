import torchxrayvision as xrv
import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np


def make_fig(plot_matrix):
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(plot_matrix, cmap=plt.cm.hot)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return fig


def predict(image, model_choice=None, model=None):
    """Function that serves predictions."""
    img = xrv_prepare_image(image)
    if model is None:
        model = xrv.models.DenseNet(weights=model_choice)
    model.eval()

    outputs = model(img)

    scores = outputs[0].detach().cpu().numpy().astype(float)
    label = dict(zip(model.pathologies, scores))
    return label


def xrv_prepare_image(image):
    img = xrv.datasets.normalize(
        image, 255
    )  # convert 8-bit image to [-1024, 1024] range
    img = img.mean(2)[None, ...] if img.ndim > 2 else img[None, ...]
    transform = torchvision.transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
    )
    img = transform(img)
    img = torch.from_numpy(img)

    return img[None, ...]
