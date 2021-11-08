from pathlib import Path
from typing import Optional, Union

import numpy as np
import timm
import torch
import torch.nn as nn


class GazeEstimator(nn.Module):
    def __init__(
        self,
        backbone_name: str = "restnet18",
        pretrained: bool = True,
        freeze: bool = True,
        model_path: Optional[Union[str, Path]] = None,
        device: Union[int, str, torch.device] = "cpu",
        cache: bool = True,
    ):
        super().__init__()

        self.device = torch.device(device)

        self.model = timm.create_model(model_name=backbone_name, pretrained=pretrained, num_classes=2)
        if pretrained:
            self.model = self.load_model(model=self.model, path=model_path, cache=cache)
        """
        if freeze:
            for param in self.model.features.parameters():
                param.requires_grad = False
        """
        self.model.to(self.device)

    def load_model(
        self,
        model: nn.Module,
        path: Optional[Union[str, Path]] = None,
        cache: bool = True,
    ) -> nn.Module:
        if path is None and cache:
            path = Path("./.cache/models/eth-xgaze_resnet18.pth")
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                torch.hub.download_url_to_file(
                    url="https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.2.2/eth-xgaze_resnet18.pth",
                    dst=path.as_posix(),
                )
        elif path is None:
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.2.2/eth-xgaze_resnet18.pth",
                map_location="cpu",
            )

        if path is not None:
            state_dict = torch.load(str(path), map_location="cpu")

        model.load_state_dict(state_dict=state_dict["model"])

        return model

    def forward(self, image: np.ndarray) -> tuple:
        return self.model(image)
