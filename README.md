# LPIPS + SSIM Image Filter

This tool automatically filters generated images using LPIPS and SSIM similarity metrics against real images. It is particularly useful for cleaning up synthetic data used in tasks like medical image segmentation.

## 📁 Project Structure

```

lpips-ssim-image-filter/
├── data/
│   ├── raw/         # Real reference images
│   ├── ema/         # Generated images
│   └── mask/        # Corresponding masks (matched by filename pattern)
├── output/          # Automatically created result folder
│   └── output\output_lpips_03_ssim_06/ # example.
├── filter\_images.py
├── run.sh
├── LICENSE
└── README.md

````

## 🧠 Features

- Compare each generated image to all real images using LPIPS and SSIM
- Retain high-quality images in `filtered_good` and discard low-quality ones in `filtered_bad`
- Automatically copy matching masks if available
- Optional LPIPS heatmap visualization
- Output includes:
  - `filtering_report.csv`
  - `result.txt` (summary)
  - `result.json` (machine-readable statistics)

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
````

### 2. Run the script

```bash
python filter_images.py \
  --real_dir data/raw \
  --gen_dir data/ema \
  --mask_dir data/mask \
  --lpips 0.3 \
  --ssim 0.6 \
  --visualize
```

### 3. Outputs

Saved to:

```
output/output_lpips_03_ssim_06/
```

Includes:

* `filtered_good/`, `filtered_bad/` image sets
* Corresponding `*_masks/` folders
* `result.txt`, `result.json`, `filtering_report.csv`

## 🔧 Parameters

| Argument         | Description                                      |
| ---------------- | ------------------------------------------------ |
| `--lpips`        | LPIPS threshold (lower = better)                 |
| `--ssim`         | SSIM threshold (higher = better)                 |
| `--size`         | Resize size for evaluation, e.g. `256,256`       |
| `--visualize`    | Save LPIPS difference heatmaps                   |
| `--batch_folder` | Optional: batch mode for multiple subdirectories |

## 📊 Result Example

`result.txt`:

```
[Result Summary]
filtered_good: 120 images
filtered_good_masks: 120 masks
filtered_bad: 30 images
filtered_bad_masks: 30 masks
avg_lpips: 0.2173
avg_ssim: 0.6851
```

`result.json`:

```json
{
  "filtered_good": 120,
  "filtered_good_masks": 120,
  "filtered_bad": 30,
  "filtered_bad_masks": 30,
  "avg_lpips": 0.2173,
  "avg_ssim": 0.6851
}
```

## 📜 License

MIT License



