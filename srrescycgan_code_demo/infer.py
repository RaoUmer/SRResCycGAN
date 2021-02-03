import time
import sys
import os
from glob import glob
import argparse
import torch
import cv2
import numpy as np

from models.ResDNet import ResDNet
from models.SRResDNet import SRResDNet
from test_srrescycgan import crop_forward

MODEL_PATHS = {
    "jpeg-compression": "srrescycgan.jpeg-compression.pth",
    "real-image-corruptions": "srrescycgan.real-image-corruptions.pth",
    "sensor-noise": "srrescycgan.sensor-noise.pth",
    "unknown-compressions": "srrescycgan.unknown-compressions.pth",
}


def main():
    parser = argparse.ArgumentParser(
        description="Deep Cyclic Generative Adversarial Residual Convolutional Networks for Real Image Super-Resolution"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="jpeg-compression",
        choices=MODEL_PATHS.keys(),
        help="Model variant to use",
    )
    parser.add_argument("--input-folder", "-i", help="Folder containing input images")
    parser.add_argument(
        "--output-folder", "-o", help="Folder to put resulting output images"
    )
    parser.add_argument("--gpu", "-g", action="store_true", help="Use GPUs")
    parser.add_argument(
        "--no-chop", action="store_true", help="Don't chop the image (uses more memory)"
    )
    args = parser.parse_args()

    if args.gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    input_folder = os.path.abspath(os.path.expanduser(args.input_folder))
    if not os.path.exists(input_folder):
        sys.stderr.write(f"Input folder does not exist: {input_folder}\n")
        sys.exit(1)

    output_folder = os.path.abspath(os.path.expanduser(args.output_folder))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_path = os.path.join("trained_nets_x4", MODEL_PATHS[args.model])
    scale = 4
    resdnet = ResDNet(depth=5)
    model = SRResDNet(resdnet, scale=scale)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    for input_path in glob(input_folder + "/*"):
        sys.stdout.write(f"Processing {input_path}...")
        sys.stdout.flush()
        try:
            base = os.path.splitext(os.path.basename(input_path))[0]
            img_data = cv2.imread(input_path, cv2.IMREAD_COLOR)
            sys.stdout.write(".")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"Failed to read {input_path}: {e}\n")
            continue

        img = torch.from_numpy(
            np.transpose(img_data[:, :, [2, 1, 0]], (2, 0, 1))
        ).float()
        img = img.unsqueeze(0)
        img = img.to(device)

        with torch.no_grad():
            start = time.time()
            try:
                if args.no_chop:
                    output_raw = model(img)
                else:
                    output_raw = crop_forward(model, img, sf=scale)
            except Exception as e:
                sys.stderr.write(f"Failed to process {input_path}: {e}\n")
                continue
            duration = time.time() - start

        sys.stdout.write(".")
        sys.stdout.flush()
        output = output_raw.data.squeeze().float().cpu().clamp_(0, 255).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))

        output_path = os.path.join(output_folder, f"{base}.png")
        try:
            cv2.imwrite(output_path, output)
        except Exception as e:
            sys.stderr.write(f"Failed to write {output_path}: {e}\n")
            continue

        del img_data, img, output_raw, output
        if args.gpu:
            torch.cuda.empty_cache()

        print(f" wrote {output_path} (took {duration:.2f} seconds)")


if __name__ == "__main__":
    main()
