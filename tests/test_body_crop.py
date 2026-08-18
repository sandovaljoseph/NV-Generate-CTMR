from __future__ import annotations

import torch

from scripts.body_crop import crop_img_body_mask, fill_body_envelope_holes


class _MetadataTensor(torch.Tensor):
    meta: dict[str, str]

    @staticmethod
    def __new__(cls, data: torch.Tensor, meta: dict[str, str]) -> _MetadataTensor:
        tensor = torch.Tensor._make_subclass(cls, data, data.requires_grad)
        tensor.meta = meta
        return tensor

    def clone(self, *args, **kwargs) -> _MetadataTensor:
        cloned = super().clone(*args, **kwargs)
        cloned.meta = self.meta.copy()
        return cloned


def test_body_crop_preserves_tensor_contract_and_fills_only_enclosed_holes() -> None:
    labels = _MetadataTensor(
        torch.ones((1, 1, 7, 7, 7), dtype=torch.int64),
        {"source": "body-mask"},
    )
    labels[:, :, 3, 3, :] = 0
    labels[:, :, 0, :, :] = 0
    image = _MetadataTensor(
        torch.full(labels.shape, 42.0, dtype=torch.float32),
        {"source": "synthetic-ct"},
    )

    filled_labels = fill_body_envelope_holes(labels)
    cropped = crop_img_body_mask(image, filled_labels)

    assert filled_labels.shape == labels.shape
    assert filled_labels.dtype == torch.int64
    assert filled_labels.meta == {"source": "body-mask"}
    assert filled_labels[0, 0, 3, 3, 3].item() == 200
    assert filled_labels[0, 0, 0, 3, 3].item() == 0
    assert cropped is image
    assert cropped.shape == image.shape
    assert cropped.dtype == torch.float32
    assert cropped.meta == {"source": "synthetic-ct"}
    assert cropped[0, 0, 3, 3, 3].item() == 42.0
    assert cropped[0, 0, 0, 3, 3].item() == -1000.0
