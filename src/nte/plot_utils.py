import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import os
import torch
import cv2
from PIL import Image
import io
import random

from sklearn.manifold import TSNE

from scipy import signal

class PlotUtils:
   
    @staticmethod 
    def apply_jpeg_compression_to_image(image, quality=100):
        img = Image.fromarray(image, "RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", quality = quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        return np.array(compressed_img)

    @staticmethod
    def apply_resize_to_image(image, scale=1.0):
        if scale >= 1.0:
            return image.copy()
    
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        y_indices = (np.arange(new_h) / scale).astype(int)
        x_indices = (np.arange(new_w) / scale).astype(int)
        resized = image[y_indices[:, np.newaxis], x_indices]
        
        padded = np.zeros_like(image)
        y_offset, x_offset = (h - new_h) // 2, (w - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded

    @staticmethod
    def apply_crop_to_image(image, percent=0.0):
        if percent <= 0.0:
            return image.copy()
    
        h, w = image.shape[:2]
        
        crop_h = int(h * percent)
        crop_w = int(w * percent)
        
        crop_h = max(1, min(crop_h, h))
        crop_w = max(1, min(crop_w, w))
        
        y_start = random.randint(0, h - crop_h)
        x_start = random.randint(0, w - crop_w)
        
        result = image.copy()
        result[y_start:y_start+crop_h, x_start:x_start+crop_w] = 0
        
        return result
    
    def apply_illumination_shifts(image: np.ndarray, strength=0.0) -> np.ndarray:
        if strength <= 0.0:
            return image.copy()

        is_float = np.issubdtype(image.dtype, np.floating)

        if is_float:
            shift = np.random.uniform(-strength, strength)
            min_bound, max_bound = 0.0, 1.0
        else:
            i_info = np.iinfo(image.dtype)
            max_int_range = i_info.max
            shift = np.random.uniform(-strength, strength) * max_int_range
            min_bound, max_bound = i_info.min, i_info.max

        shifted_image = image.astype(np.float32) + shift
        shifted_image = np.clip(shifted_image, min_bound, max_bound)
        return shifted_image.astype(image.dtype)
    
    @staticmethod
    def show_image_comparison(img_before: np.ndarray, img_after: np.ndarray, title_before: str = "Before", title_after: str = "After", filename: Optional[str] = None):
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(img_before, cmap='gray' if img_before.ndim == 2 else None)
        axs[0].set_title(title_before)
        axs[0].axis('off')
        axs[1].imshow(img_after, cmap='gray' if img_after.ndim == 2 else None)
        axs[1].set_title(title_after)
        axs[1].axis('off')
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    def show_image(img, processor):
        img = PlotUtils.deprocess_func(img, processor)

    @staticmethod
    def deprocess_func(img: torch.Tensor, processor) -> np.ndarray:
        mean = torch.tensor(processor.image_mean).view(3, 1, 1)
        std = torch.tensor(processor.image_std).view(3, 1, 1)
        tensor = img * std + mean
        tensor = tensor.clamp(0, 1) * 255
        tensor = tensor.byte()
        
        if tensor.dim() == 3 and tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        
        return tensor.cpu().numpy()

    @staticmethod
    def show_processed_image_comparison(img_before: torch.Tensor, img_after: torch.Tensor, processor, title_before: str = "Before", title_after: str = "After", filename: Optional[str] = None):
        processed_before = PlotUtils.deprocess_func(img_before, processor)
        processed_after = PlotUtils.deprocess_func(img_after, processor)
        PlotUtils.show_image_comparison(processed_before, processed_after, title_before, title_after, filename)

    
    @staticmethod
    def show_embeddings_tsne(emb_before: np.ndarray, emb_after: np.ndarray, label_before: str = "Before", label_after: str = "After", tokens=None, random_state: Optional[int] = 42, filename: Optional[str] = None):
        X = np.concatenate([emb_before, emb_after], axis=0)
        y = np.array([label_before] * len(emb_before) + [label_after] * len(emb_after))
        tsne = TSNE(n_components=2, random_state=random_state, perplexity=5)
        X_2d = tsne.fit_transform(X)
        plt.figure(figsize=(6, 6))
        for label, color in zip([label_before, label_after], ['blue', 'red']):
            idx = y == label
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=label, alpha=0.6, s=20, c=color)
        if tokens is not None:
            for i, token in enumerate(tokens):
                plt.annotate(token, (X_2d[i, 0], X_2d[i, 1]), fontsize=8, alpha=0.6)
                plt.annotate(token, (X_2d[i+len(tokens), 0], X_2d[i+len(tokens), 1]), fontsize=8, alpha=0.6, color='red')
        plt.legend()
        plt.tight_layout()
        plt.show()
        if filename is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
        else:
            plt.show()


    @staticmethod
    def show_audio_spectrogram(wav_before: np.ndarray, wav_after: np.ndarray, sr: int, n_fft: int = 512, hop_length: int = 256, title_before: str = "Before", title_after: str = "After"):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for i, (wav, title, ax) in enumerate(zip([wav_before, wav_after], [title_before, title_after], axs)):
            f, t, Sxx = signal.spectrogram(wav, sr, nperseg=n_fft, noverlap=n_fft - hop_length)
            ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
            ax.set_title(title)
            ax.set_ylabel('Frequency [Hz]')
            ax.set_xlabel('Time [sec]')
        plt.tight_layout()
        plt.show()
