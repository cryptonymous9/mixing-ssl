# Image Generation with Diffusion Models

Generate images using Stable Diffusion or Versatile Diffusion models from an input dataset.

## Usage 

Run the script using the following command:

```bash
python run.py \
  --data_dir /path/to/data \
  --output_dir /path/to/output \
  --steps 50 \
  --batch_size 16 \ 
  --guidance_scale 3.0 \
  --model sd
```

## Dataset Structure

The input dataset should be organized as follows:

```
data_dir/
  class1/
    image1.jpg
    image2.jpg
    ...
  class2/
    image1.jpg
    image2.jpg
    ...
```
Each subdirectory corresponds to a label (e.g., class1, class2). Images are stored inside their respective label subdirectories.
