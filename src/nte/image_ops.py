import io
import random
import numpy as np
import torch
from PIL import Image
import cv2


class ImgOps:

    @staticmethod
    def _to_numpy_bhwc(x: torch.Tensor):
        assert isinstance(x, torch.Tensor), "Only torch.Tensor is supported"

        device = x.device
        dtype = x.dtype
        arr = x.detach().cpu().numpy()

        added_batch = False
        if arr.ndim == 3:
            arr = arr[None, ...]
            added_batch = True
        elif arr.ndim != 4:
            raise ValueError(f"Expect 3D or 4D tensor, got {arr.ndim}D")

        chw = False
        if arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (0, 2, 3, 1))
            chw = True

        return arr, chw, added_batch, dtype, device

    @staticmethod
    def _from_numpy_bhwc(arr, chw, added_batch, dtype, device):
        if chw:
            arr = np.transpose(arr, (0, 3, 1, 2))

        if added_batch:
            arr = arr[0]

        t = torch.from_numpy(arr).to(device)
        return t.to(dtype)

    @staticmethod
    def apply_png_compression_to_image(image: torch.Tensor, threshold=1):
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")
        if image.dim() != 4:
            raise ValueError("image must be 4D tensor (B, C, H, W)")

        device = image.device
        dtype = image.dtype

        x = image.detach().cpu().permute(0, 2, 3, 1)
        B, H, W, C = x.shape

        x_flat = x.reshape(B, -1)
        mins = x_flat.min(dim=1).values.view(B, 1, 1, 1)
        maxs = x_flat.max(dim=1).values.view(B, 1, 1, 1)
        scale = (maxs - mins).clamp(min=1e-6)

        normed = (x - mins) / scale
        normed = (normed * 255.0).clamp(0.0, 255.0).round().to(torch.uint8)

        outs = []
        compress_level = int(np.clip(threshold, 0, 9))
        for i in range(B):
            arr = normed[i].numpy()
            img = Image.fromarray(arr, mode="RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG", compress_level=compress_level)
            buf.seek(0)
            reloaded = np.array(Image.open(buf).convert("RGB"))
            outs.append(reloaded)

        out = np.stack(outs, axis=0)
        out = torch.from_numpy(out).float() / 255.0

        out = out * scale + mins

        out = out.permute(0, 3, 1, 2).to(device=device, dtype=dtype)
        return out

    @staticmethod
    def apply_jpeg_compression_to_image(image: torch.Tensor, threshold=1):
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")
        if image.dim() != 4:
            raise ValueError("image must be 4D tensor (B, C, H, W)")

        device = image.device
        dtype = image.dtype

        x = image.detach().cpu().permute(0, 2, 3, 1)
        B, H, W, C = x.shape

        x_flat = x.reshape(B, -1)
        mins = x_flat.min(dim=1).values.view(B, 1, 1, 1)
        maxs = x_flat.max(dim=1).values.view(B, 1, 1, 1)
        scale = (maxs - mins).clamp(min=1e-6)

        normed = (x - mins) / scale
        normed = (normed * 255.0).clamp(0.0, 255.0).round().to(torch.uint8)

        outs = []
        for i in range(B):
            arr = normed[i].numpy()
            img = Image.fromarray(arr, mode="RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=int(threshold))
            buf.seek(0)
            reloaded = np.array(Image.open(buf).convert("RGB"))
            outs.append(reloaded)

        out = np.stack(outs, axis=0)
        out = torch.from_numpy(out).float() / 255.0

        out = out * scale + mins

        out = out.permute(0, 3, 1, 2).to(device=device, dtype=dtype)
        return out


    @staticmethod
    def apply_resize_to_image(image: torch.Tensor, threshold=1.0):
        arr, chw, added_batch, dtype, device = ImgOps._to_numpy_bhwc(image)
        B, H, W, C = arr.shape

        if threshold == 1.0:
            return ImgOps._from_numpy_bhwc(arr.copy(), chw, added_batch, dtype, device)

        threshold = max(0.0, min(2.0, float(threshold)))
        if threshold == 0.0:
            return ImgOps._from_numpy_bhwc(np.zeros_like(arr), chw, added_batch, dtype, device)

        new_h = max(1, int(H * threshold))
        new_w = max(1, int(W * threshold))

        resized = np.empty((B, new_h, new_w, C), dtype=arr.dtype)
        for b in range(B):
            resized[b] = cv2.resize(
                arr[b], (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )

        resized_back = np.empty_like(arr)
        for b in range(B):
            resized_back[b] = cv2.resize(
                resized[b], (W, H), interpolation=cv2.INTER_LINEAR
            )

        return ImgOps._from_numpy_bhwc(resized_back, chw, added_batch, dtype, device)

    @staticmethod
    def apply_pad_resize_to_image(image: torch.Tensor, threshold=1.0):
        arr, chw, added_batch, dtype, device = ImgOps._to_numpy_bhwc(image)
        B, H, W, C = arr.shape

        if threshold >= 1.0:
            return ImgOps._from_numpy_bhwc(arr.copy(), chw, added_batch, dtype, device)

        new_h, new_w = int(H * threshold), int(W * threshold)
        y_idx = (np.arange(new_h) / threshold).astype(int)
        x_idx = (np.arange(new_w) / threshold).astype(int)

        resized = arr[:, y_idx[:, None], x_idx, :]

        padded = np.zeros_like(arr)
        y0 = (H - new_h) // 2
        x0 = (W - new_w) // 2
        padded[:, y0:y0 + new_h, x0:x0 + new_w, :] = resized

        return ImgOps._from_numpy_bhwc(padded, chw, added_batch, dtype, device)

    @staticmethod
    def apply_crop_to_image(image: torch.Tensor, threshold=0.0):
        arr, chw, added_batch, dtype, device = ImgOps._to_numpy_bhwc(image)

        if threshold <= 0.0:
            return ImgOps._from_numpy_bhwc(arr.copy(), chw, added_batch, dtype, device)

        B, H, W, C = arr.shape
        ch = max(1, int(H * threshold))
        cw = max(1, int(W * threshold))

        out = arr.copy()
        for i in range(B):
            y = random.randint(0, H - ch)
            x = random.randint(0, W - cw)
            out[i, y:y + ch, x:x + cw, :] = 0

        return ImgOps._from_numpy_bhwc(out, chw, added_batch, dtype, device)

    @staticmethod
    def apply_illumination_shifts(image: torch.Tensor, threshold=0.0):
        arr, chw, added_batch, dtype, device = ImgOps._to_numpy_bhwc(image)

        if threshold <= 0.0:
            return ImgOps._from_numpy_bhwc(arr.copy(), chw, added_batch, dtype, device)

        is_float = np.issubdtype(arr.dtype, np.floating)
        B = arr.shape[0]

        shifts = np.random.uniform(-threshold, threshold, size=(B, 1, 1, 1))
        out = arr + shifts

        return ImgOps._from_numpy_bhwc(out, chw, added_batch, dtype, device)
