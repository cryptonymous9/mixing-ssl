import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import (
    VersatileDiffusionImageVariationPipeline,
    StableDiffusionImageVariationPipeline
)
from diffusers.pipelines.stable_diffusion import safety_checker
from tqdm.auto import tqdm
from PIL import Image

# Override the safety checker
def disable_safety_checker(self, clip_input, images):
    return images, [False for _ in images]
safety_checker.StableDiffusionSafetyChecker.forward = disable_safety_checker


class ImageDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        """
        Custom dataset to load images from directory structure.

        Args:
            data_dir (str): Path to the dataset root directory.
            transform (callable, optional): Transformations to apply to the images. Defaults to None.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []

        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):  # Skip non-directory files
                continue
            for image_name in os.listdir(class_path):
                if image_name.startswith('.'):  # Skip hidden files and folders
                    continue
                self.image_paths.append(os.path.join(class_name, image_name))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = os.path.join(self.data_dir, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label, file_name = self.image_paths[idx].split("/")
        return image, label, file_name


def generate_images(
    pipeline, output_dir: str, generator, dataloader, guidance_scale: float, num_steps: int
):
    """
    Generate images using a diffusion pipeline.

    Args:
        pipeline: Diffusion pipeline instance.
        output_dir (str): Directory to save generated images.
        generator: Random generator for reproducibility.
        dataloader: Dataloader with input images.
        guidance_scale (float): Guidance scale for the pipeline.
        num_steps (int): Number of inference steps.
    """
    for _, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        images, labels, file_names = data
        images = images.to(device)
        results = pipeline(
            images,
            generator=generator,
            height=512,
            width=512,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
        ).images

        for result, label, file_name in zip(results, labels, file_names):
            save_dir = os.path.join(output_dir, label)
            os.makedirs(save_dir, exist_ok=True)
            result.save(os.path.join(save_dir, file_name))
    return True


def parse_arguments():
    parser = argparse.ArgumentParser(description="Image Generation with Diffusion Models")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="Guidance scale")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the input dataset")
    parser.add_argument("--image_size", type=int, default=512, help="Image size (for resizing)")
    parser.add_argument("--seed", type=int, default=25, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for dataloader")
    parser.add_argument("--model", choices=["sd", "ver"], default="sd", help="Model type")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.model == "ver":
        print("Using model: shi-labs/versatile-diffusion")
        pipeline = VersatileDiffusionImageVariationPipeline.from_pretrained(
            "shi-labs/versatile-diffusion", torch_dtype=torch.float16
        )
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (args.image_size, args.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=False,
            ),
            transforms.Lambda(lambda x: (x * 255).byte()),
        ])
    else:
        print("Using model: lambdalabs/sd-image-variations-diffusers")
        pipeline = StableDiffusionImageVariationPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers",
            revision="v2.0",
            torch_dtype=torch.float16,
        )
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=False,
            ),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"Guidance scale: {args.guidance_scale}, Starting!")

    dataset = ImageDataset(
        data_dir=args.data_dir,
        transform=transform,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    generate_images(
        pipeline, args.output_dir, generator, dataloader, args.guidance_scale, args.steps
    )

    print(f"Guidance scale: {args.guidance_scale}, Finished!")
