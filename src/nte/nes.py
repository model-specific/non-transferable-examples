import torch
from torch import nn
import torch.nn.functional as F
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NES:

    def __init__(self, model: nn.Module=None, tau=1e-5):
        self.model = model
        self.tau = tau

    def generate_noise_vector(self, mode="first_layer", layer_name=None, layer=None):
        if not self.model:
            raise ValueError("Model is not defined.")
        if mode not in ["first_layer", "jacobian"]:
            raise ValueError("Mode must be 'first_layer' or 'jacobian'.")
        if mode == "first_layer":
            if layer is None:
                if layer_name is None:
                    raise ValueError("Layer name must be provided for 'first_layer' mode.")
                layer = dict(self.model.named_modules()).get(layer_name, None)
            if layer is None:
                raise ValueError(f"Layer '{layer_name}' not found in the model.")
            if isinstance(layer, nn.Conv2d):
                logger.info(f"Extract Conv Null Space. conv={layer}")
                return self.extract_null_space_vectors(layer, self.tau), layer
            if isinstance(layer, nn.Linear):
                pass
            raise ValueError(f"Unsupport Layer Type. Layer '{layer_name}' should be Conv2d or Linear.") 
        elif mode == "jacobian":
            raise NotImplementedError("Jacobian mode is not implemented yet.")
        else:
            raise ValueError("Invalid mode. Choose 'first_layer' or 'jacobian'.")
        
    def extract_null_space_vectors(self, layer, tau):
        w_flat = layer.weight.view(layer.out_channels, -1)
        with torch.no_grad():
            cost_time = time.time()
            _, s, vh = torch.linalg.svd(w_flat)
            rank = s.size(0) if vh.size(0) > s.size(0) else (s > tau).sum().item()
            null_space_vectors = vh[rank:].T
            logger.info(f"Extract null space vectors. cost_time={time.time() - cost_time}, w.shape={layer.weight.shape}, w_flat.shape={w_flat.shape}, rank={rank}, s.shape={s.shape}, vh.shape={vh.shape}. s[-5:]={s[-5:]}")
            if null_space_vectors.size(1) == 0:
                logger.warning(f"No null space found; using minimal column space vector instead. tau={self.tau}, S=...{s[-5:]}") 
                null_space_vectors = vh[-1:].T
        return null_space_vectors

    def generate_cnn_noised_images(self, images, null_space_vectors, psnr, noise_type="nes", kernel_size=None, stride=None, padding=None, compute_cost=False):
        cost_time = 0
        if kernel_size is None or stride is None or padding is None:
            raise ValueError("kernel_size, stride, and padding must be specified for CNN noise addition.")
        if padding == "valid" or type(padding) == "str":
            padding = 0 
        if images.dim() != 4:
            raise ValueError("Input images must be a 4D tensor with shape (B, C, H, W).")
        if (stride and kernel_size) and stride[0] < kernel_size[0]:
            pass
        unfolded_images = F.unfold(images, kernel_size=kernel_size, stride=stride, padding=padding)
        if noise_type == "nes":
            start_time = time.perf_counter_ns() 
            z = torch.randn(images.shape[0], null_space_vectors.shape[-1], 1, device=images.device)
            noise = torch.einsum("ij, bjl->bil", null_space_vectors, z)
            cost_time += time.perf_counter_ns() - start_time
        elif noise_type == "gaussian":
            noise = torch.randn_like(unfolded_images)
        else:
            raise ValueError("Unsupport noise_type.")

        target_mse = (255 ** 2) / (10 ** (psnr/10))

        noised_unfolded_images = unfolded_images + noise
        noise_mse = torch.mean(noised_unfolded_images ** 2, dim=(1, 2), keepdim=True)
        scale = torch.sqrt(target_mse / noise_mse) if torch.any(noise_mse > 0) else 1.0
        if noise_type == "nes":
            z = torch.randn(images.shape[0], null_space_vectors.shape[-1], 1, device=images.device)
            scaled_noise = torch.einsum("ij, bjl->bil", null_space_vectors, z * scale)

        else:
            scaled_noise = scale * noise
        overlap_count = torch.ones_like(unfolded_images)
        overlap_count = F.fold(overlap_count, output_size=images.shape[-2:], kernel_size=kernel_size, stride=stride, padding=padding)
        overlap_count[overlap_count > 1] = 0
        unfold_overlap_count = F.unfold(overlap_count, kernel_size=kernel_size, stride=stride, padding=padding)
        start_time = time.time()
        noised_images = unfolded_images + scaled_noise 
        cost_time += time.time() - start_time
        folded_noised = F.fold(noised_images, output_size=images.shape[-2:], kernel_size=kernel_size, stride=stride, padding=padding)
        logger.info(f"cost_time={cost_time}, overlap_cnt={unfold_overlap_count.sum()}, scale={scale.max().item()}, max(images)={images.max().item()}, noised_images={noised_images.max().item()}, max(noise)={noise.max().item()}, max(folded_scaled_noised)={folded_noised.max().item()}, min(overlap): {overlap_count.min().item()}")
        return folded_noised, noised_images


    
    def calculate_psnr(self, original, noisy):
        if original.shape != noisy.shape:
            raise ValueError("Original and noisy images must have the same shape")
        
        mse = torch.mean((original - noisy) ** 2)
        max_pixel = 255
        if mse == 0:
            return float('inf')
            
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.mean().item() if psnr.numel() > 1 else psnr.item()
            
    def add_linear_noise(self, inputs, null_space_vectors, alpha):
        pass
    
if __name__ == "__main__":

    from transformers import AutoModelForImageClassification, AutoProcessor
    from PIL import Image
    model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch17-224")
    processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224")
    nes = NES(model=model, tau=1e-4)
    nsv, layer = nes.generate_noise_vector(layer_name="vit.embeddings.patch_embeddings.projection")
    x = Image.open("data/Lenna.png").convert("RGB")
    x = processor(images=x, return_tensors="pt")["pixel_values"]

    noised_images = nes.generate_cnn_noised_images(x, nsv, psnr=19, kernel_size=layer.kernel_size, stride=layer.stride, padding= layer.padding)
    psnr = nes.calculate_psnr(x, noised_images)
    print(f"PSNR={psnr}")
