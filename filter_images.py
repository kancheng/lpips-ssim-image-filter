import os
import shutil
import argparse
import torch
import lpips
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import json


def visualize_lpips_diff(gen_img, real_img, diff_map, output_path):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(gen_img)
    axs[0].set_title("Generated")
    axs[1].imshow(real_img)
    axs[1].set_title("Closest Real")
    axs[2].imshow(diff_map, cmap='hot')
    axs[2].set_title("LPIPS Diff")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def process_one_folder(real_dir, gen_dir, mask_dir, output_dir, lpips_model, args):
    transform = transforms.Compose([
        transforms.Resize(args.size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/filtered_good", exist_ok=True)
    os.makedirs(f"{output_dir}/filtered_bad", exist_ok=True)
    os.makedirs(f"{output_dir}/filtered_good_masks", exist_ok=True)
    os.makedirs(f"{output_dir}/filtered_bad_masks", exist_ok=True)
    if args.visualize:
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)

    real_images = []
    for f in os.listdir(real_dir):
        try:
            img = Image.open(os.path.join(real_dir, f)).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            real_images.append((f, tensor, img))
        except:
            continue

    records = []
    for fname in tqdm(os.listdir(gen_dir)):
        try:
            gen_img = Image.open(os.path.join(gen_dir, fname)).convert('RGB')
            gen_tensor = transform(gen_img).unsqueeze(0).to(device)

            best_lpips, best_ssim = float('inf'), 0
            best_real_img = None
            best_lpips_map = None

            for rf, r_tensor, r_img in real_images:
                d = lpips_model(gen_tensor, r_tensor)
                s = ssim(np.array(gen_img.resize(args.size).convert('L')) / 255.0,
                         np.array(r_img.resize(args.size).convert('L')) / 255.0,
                         data_range=1.0)
                if d.item() < best_lpips:
                    best_lpips = d.item()
                    best_real_img = r_img
                    best_lpips_map = d.squeeze().detach().cpu().numpy()
                best_ssim = max(best_ssim, s)

            records.append({'filename': fname, 'lpips': best_lpips, 'ssim': best_ssim})

            # 複製圖像與 mask
            mask_file = fname.replace('_ema', '_mask')
            src_img = os.path.join(gen_dir, fname)
            src_mask = os.path.join(mask_dir, mask_file)
            if best_lpips < args.lpips and best_ssim > args.ssim:
                shutil.copy(src_img, os.path.join(output_dir, "filtered_good", fname))
                if os.path.exists(src_mask):
                    shutil.copy(src_mask, os.path.join(output_dir, "filtered_good_masks", mask_file))
            else:
                shutil.copy(src_img, os.path.join(output_dir, "filtered_bad", fname))
                if os.path.exists(src_mask):
                    shutil.copy(src_mask, os.path.join(output_dir, "filtered_bad_masks", mask_file))

            # 可視化
            if args.visualize and best_real_img is not None:
                visualize_lpips_diff(gen_img, best_real_img, best_lpips_map,
                    os.path.join(output_dir, "visualizations", f"diff_{fname}.png"))
        except Exception as e:
            print(f"Error {fname}: {e}")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "filtering_report.csv"), index=False)

    # 統計結果儲存為 result.txt 與 result.json
    count_good = len(os.listdir(os.path.join(output_dir, 'filtered_good')))
    count_good_mask = len(os.listdir(os.path.join(output_dir, 'filtered_good_masks')))
    count_bad = len(os.listdir(os.path.join(output_dir, 'filtered_bad')))
    count_bad_mask = len(os.listdir(os.path.join(output_dir, 'filtered_bad_masks')))
    avg_lpips = df['lpips'].mean()
    avg_ssim = df['ssim'].mean()

    result_path = os.path.join(output_dir, "result.txt")
    with open(result_path, 'w') as f:
        f.write("[Result Summary]\n")
        f.write(f"filtered_good: {count_good} images\n")
        f.write(f"filtered_good_masks: {count_good_mask} masks\n")
        f.write(f"filtered_bad: {count_bad} images\n")
        f.write(f"filtered_bad_masks: {count_bad_mask} masks\n")
        f.write(f"avg_lpips: {avg_lpips:.4f}\n")
        f.write(f"avg_ssim: {avg_ssim:.4f}\n")

    json_result = {
        "filtered_good": count_good,
        "filtered_good_masks": count_good_mask,
        "filtered_bad": count_bad,
        "filtered_bad_masks": count_bad_mask,
        "avg_lpips": avg_lpips,
        "avg_ssim": avg_ssim
    }
    with open(os.path.join(output_dir, "result.json"), 'w') as jf:
        json.dump(json_result, jf, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, default="./data/raw/")
    parser.add_argument("--gen_dir", type=str, default="./data/ema/")
    parser.add_argument("--mask_dir", type=str, default="./data/mask/")
    parser.add_argument("--output_dir", type=str, default="./data/")
    parser.add_argument("--lpips", type=float, default=0.3)
    parser.add_argument("--ssim", type=float, default=0.6)
    parser.add_argument("--size", type=lambda s: tuple(map(int, s.split(','))), default=(256, 256))
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--batch_folder", type=str, default=None)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net='alex').to(device)

    # 修改 output 資料夾名稱加上參數識別
    lpips_str = str(args.lpips).replace('.', '')
    ssim_str = str(args.ssim).replace('.', '')
    args.output_dir = os.path.join("output", f"output_lpips_{lpips_str}_ssim_{ssim_str}")

    if args.batch_folder:
        subdirs = sorted(os.listdir(args.batch_folder))
        for sub in subdirs:
            r = os.path.join(args.batch_folder, sub, "real")
            g = os.path.join(args.batch_folder, sub, "gen")
            m = os.path.join(args.batch_folder, sub, "mask")
            o = os.path.join(args.batch_folder, sub, "output")
            print(f"\n📁 Processing {sub}")
            process_one_folder(r, g, m, o, lpips_model, args)
    else:
        process_one_folder(args.real_dir, args.gen_dir, args.mask_dir, args.output_dir, lpips_model, args)

