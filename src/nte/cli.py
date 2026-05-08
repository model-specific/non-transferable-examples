import sys
import os
import argparse
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, TrainingArguments, Trainer, AutoModelForImageClassification, AutoImageProcessor
from transformers.models.resnet.modeling_resnet import ResNetForImageClassification
from datasets import load_dataset
import evaluate
import numpy as np
from PIL import Image
from .nes import NES
from .adapters import CustomResNetEmbedding, CustomPatchEmbed
from .plot_utils import PlotUtils
from .image_ops import ImgOps
import random

os.environ["WANDB_DISABLED"] = "true"
model_names = {
    "apple/mobilevit-small": "mobilevit.conv_stem.convolution",
    "timm/resnext50_32x4d.a1h_in1k": "timm_model.conv1",
    "microsoft/resnet-50": "resnet.embedder.embedder.convolution",
    "timm/inception_v3.tv_in1k": "timm_model.Conv2d_1a_3x3.conv",
    "timm/mobilenetv4_conv_small_050.e3000_r224_in1k": "timm_model.conv_stem",
    "timm/convnext_base.fb_in22k_ft_in1k": "timm_model.stem.0",
    "facebook/convnext-large-224-22k-1k": "convnext.embeddings.patch_embeddings",
    "facebook/convnext-base-224-22k-1k": "convnext.embeddings.patch_embeddings",
    "facebook/convnext-tiny-224": "convnext.embeddings.patch_embeddings",
    "facebook/convnext-small-224": "convnext.embeddings.patch_embeddings",
    "facebook/convnext-base-224": "convnext.embeddings.patch_embeddings",
    "facebook/convnext-large-224": "convnext.embeddings.patch_embeddings",
    "microsoft/swin-base-patch4-window7-224": "swin.embeddings.patch_embeddings.projection", 
    "microsoft/swin-base-patch4-window12-384": "swin.embeddings.patch_embeddings.projection", 
    "nvidia/MambaVision-T-1K": "model.patch_embed.conv_down.0",
    "nvidia/MambaVision-B-1K": "model.patch_embed.conv_down.0",
    "nvidia/MambaVision-B-1K": "model.patch_embed.conv_down.0",
    "facebook/deit-base-patch16-224": "vit.embeddings.patch_embeddings.projection",
    "google/vit-base-patch16-224": "vit.embeddings.patch_embeddings.projection",
    "microsoft/swinv2-base-patch4-window16-256": "swinv2.embeddings.patch_embeddings.projection",
    "microsoft/swinv2-base-patch4-window8-256": "swinv2.embeddings.patch_embeddings.projection",
    "microsoft/swinv2-tiny-patch4-window8-256": "swinv2.embeddings.patch_embeddings.projection",
    "google/efficientnet-b0": "efficientnet.embeddings.convolution",
    "google/efficientnet-b1": "efficientnet.embeddings.convolution",
    "google/efficientnet-b2": "efficientnet.embeddings.convolution",
    "google/efficientnet-b5": "efficientnet.embeddings.convolution",
    "google/efficientnet-b7": "efficientnet.embeddings.convolution",
    "google/mobilenet_v2_1.0_224": "mobilenet_v2.conv_stem.first_conv.convolution"
}


def eval_vision_model(args):
    
    model_name = args.model_name
    if args.model_path is not None:
        model = AutoModelForImageClassification.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        model = AutoModelForImageClassification.from_pretrained(model_name, trust_remote_code=True)
    if model_name.startswith("nvidia/MambaVision"):
        from timm.data.transforms_factory import create_transform
        processor = create_transform(input_size=(args.batch_size, 3, 224, 224),
                             is_training=False,
                             mean=model.config.mean,
                             std=model.config.std,
                             crop_mode=model.config.crop_mode,
                             crop_pct=model.config.crop_pct)
    else:
        processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    if args.cross_model_name is not None:
        if args.cross_model_path is not None:
            cross_model = AutoModelForImageClassification.from_pretrained(args.cross_model_path, trust_remote_code=True)
        else:
            cross_model = AutoModelForImageClassification.from_pretrained(args.cross_model_name, trust_remote_code=True)
        if args.cross_model_name.startswith("nvidia/MambaVision"):
            from timm.data.transforms_factory import create_transform
            processor = create_transform(input_size=(args.batch_size, 3, 224, 224),
                                is_training=False,
                                mean=cross_model.config.mean,
                                std=cross_model.config.std,
                                crop_mode=cross_model.config.crop_mode,
                                crop_pct=cross_model.config.crop_pct)
        else:
            processor = AutoImageProcessor.from_pretrained(args.cross_model_name, trust_remote_code=True)

    params_dict = dict(model.named_modules())
    layer_name = model_names.get(model_name)
    conv1 = params_dict.get(layer_name)

    if not conv1:
        raise ValueError(f"Unsupported model: {model_name}")
    if args.noise_type=="nes" and args.cross_model_name is None:
        if model_name == "microsoft/resnet-50":
            embedder = model.resnet.embedder
            model.resnet.embedder = CustomResNetEmbedding(embedder, output_size=(-1, 64, 112, 112))
        if model_name == "nvidia/MambaVision-T-1K": 
            patch_embed = model.model.patch_embed
            custom_embed = CustomPatchEmbed(patch_embed, output_size=(-1, 32, 112, 112), in_dim=model.config.in_dim, dim=model.config.dim).to(model.device)
            model.model.patch_embed = custom_embed


    ds = load_dataset(args.dataset, split=args.split if args.split else "test", token=args.token, )
    if args.dataset == "cifar10":
        ds = ds.rename_column("img", "image")
        ds = ds.rename_column("label", "labels")
    if args.dataset == "imagenet-1k":
        ds = ds.rename_column("label", "labels")
    if args.dataset == "Nagabu/HAM10000":
        ds = ds.rename_column("label", "labels")
    if args.n_samples > 0:
        ds = ds.shuffle(seed=42).select(range(args.n_samples)) 

    def process_examples(examples):
        images = [img.convert("RGB") for img in examples["image"]]
        if "nvidia/MambaVision" in args.model_name and args.cross_model_name is None:
            processed = [processor(image) for image in images]
            inputs = {"pixel_values": torch.stack(processed)}
        elif (args.cross_model_name is not None and "nvidia/MambaVision" in args.cross_model_name):
            processed = [processor(image) for image in images]
            inputs = {"pixel_values": torch.stack(processed)}
        else:
            inputs = processor(images, return_tensors="pt")
        inputs["labels"] = examples["labels"]
        return inputs

    ds = ds.with_transform(process_examples)
    metric = evaluate.load("accuracy")

    nes = NES(model=model, tau=args.tau_w)    
    null_space_vectors, layer = nes.generate_noise_vector(mode="first_layer", layer=conv1)

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)
    
    def build_collector_fn(flat=False):
        def collector_fn(data):
            pixel_values = torch.stack([f["pixel_values"] for f in data])
            if args.noise_type == "clean":
                noised_pixel_values = pixel_values
            elif args.noise_type == "nes":
                noised_pixel_values, unfolded_pixel_values = nes.generate_cnn_noised_images(pixel_values, null_space_vectors, psnr=args.psnr, kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
            elif args.noise_type == "gaussian":
                noised_pixel_values, unfolded_pixel_values = nes.generate_cnn_noised_images(pixel_values, null_space_vectors, psnr=args.psnr, noise_type="gaussian", kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
            else:
                raise ValueError("Unsupport noise type error. It should be chosen in [clean, nes, gaussian]")

            if args.visualize is True:
                for i in range(5):
                    img = pixel_values[i].permute(1, 2, 0).detach().numpy()
                    perturbed_img = noised_pixel_values[0].permute(1, 2, 0).detach().numpy()
                    PlotUtils.show_image_comparison(img, perturbed_img, title_before="Original", title_after=f"PSNR (std={args.psnr})", filename=os.path.join(args.output_dir, f"comparison_{i}.png"))

                args.visualize=False

            threshold = args.transform_threshold
            if args.transform_type == "jpeg":
                noised_pixel_values = ImgOps.apply_jpeg_compression_to_image(noised_pixel_values, threshold)
            elif args.transform_type == "crop":
                noised_pixel_values = ImgOps.apply_crop_to_image(noised_pixel_values, threshold)
            elif args.transform_type == "resize":
                noised_pixel_values = ImgOps.apply_resize_to_image(noised_pixel_values, threshold)
            elif args.transform_type == "illumination":
                noised_pixel_values = ImgOps.apply_illumination_shifts(noised_pixel_values, threshold)
            else:
                pass

            if flat:
                return {
                    "pixel_values": unfolded_pixel_values,
                    "labels": torch.tensor([f["labels"] for f in data]),
                }

            return {
                "pixel_values": noised_pixel_values,
                "labels": torch.tensor([f["labels"] for f in data]),
            }
        return collector_fn

    model = model if args.cross_model_name is None else cross_model
    use_flat_inputs = False

    if args.noise_type == "nes" and model_name in ["microsoft/resnet-50", "nvidia/MambaVision-T-1K"]:
        if args.cross_model_name is not None:
            use_flat_inputs = False 
        else:
            use_flat_inputs = True
    return model.eval(), ds, compute_metrics, build_collector_fn(flat=use_flat_inputs) 

def main():
    parser = argparse.ArgumentParser(description="Evaluate model with null-space perturbations")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=False, default=None)
    parser.add_argument("--image_processor_name_or_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_samples", type=int, default=-1, help="Number of samples to evaluate")

    parser.add_argument("--noise_type", type=str, choices=["nes", "gaussian", "clean"], default=None)
    parser.add_argument("--psnr", type=float, required=True)
    parser.add_argument("--transform_type", type=str, choices=["jpeg", "resize", "crop", "illumination"], default=None)
    parser.add_argument("--transform_threshold", type=float, default=None)
    parser.add_argument("--tau_w", type=float, required=True, default=1e-2)
    parser.add_argument("--debug", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--cross_model_name", type=str, default=None)
    parser.add_argument("--cross_model_path", type=str, default=None)
    parser.add_argument("--visualize", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token for private models")

    args = parser.parse_args()

    model_name = args.model_name
    if model_name in model_names.keys():
        model, ds, compute_metrics, collector_fn = eval_vision_model(args)
    elif model_name in ["bert-base-uncased", "roberta-base"]:
        pass
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=True,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to=None
    )
    class MambaVisionTrainer(Trainer):
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            with torch.no_grad():
                outputs = model(inputs["pixel_values"])
                loss = None
                if "labels" in inputs:
                    logits = outputs["logits"]
                    predicted_class_idx = logits.argmax(-1).item()
                    loss = self.compute_loss(model, logits, inputs)
            return (loss, predict_class_idx, inputs.get("labels"))

    class MambaVisionWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, pixel_values=None, labels=None):
            outputs = self.model(pixel_values)
            if labels is not None:
                loss = torch.nn.CrossEntropyLoss()(outputs['logits'], labels)
                return {'loss': loss, 'logits': outputs['logits']}
            return outputs

    if "nvidia/MambaVision" in args.model_name or (args.cross_model_name is not None and "nvidia/MambaVision" in args.cross_model_name):
        model = MambaVisionWrapper(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=ds,
        compute_metrics=compute_metrics,
        data_collator=collector_fn
    )

    results = trainer.evaluate()
    trainer.log_metrics("eval", results)
    trainer.save_metrics("eval", results)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed()
    main()
