"""Figure: what the discriminator sees. Real MNIST digits under each D-input transform.

Columns: original, translation, cutout, brightness/contrast, all three, instance noise (std 0.1).
Displayed as (x+1)/2, i.e. the training normalisation inverted. Seeded, real training images only.
Usage: python scripts/diffaug_explainer_figure.py --data-path DIR --out experiments/figures/diffaugment-what-d-sees.png
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffaugment import diff_augment

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-path", type=Path, default=ROOT / "data")
parser.add_argument("--out", type=Path, default=ROOT / "experiments/figures/diffaugment-what-d-sees.png")
parser.add_argument("--n", type=int, default=6)
parser.add_argument("--seed", type=int, default=3)
args = parser.parse_args()

torch.manual_seed(args.seed)
data = datasets.MNIST(str(args.data_path), train=True, download=False,
                      transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
idx = torch.randperm(len(data))[:args.n]
x = torch.stack([data[int(i)][0] for i in idx]).view(args.n, -1)          # n x 784 in [-1, 1]
columns = [("original", x),
           ("translation", diff_augment(x, ("translation",))),
           ("cutout", diff_augment(x, ("cutout",))),
           ("brightness /\ncontrast", diff_augment(x, ("color",))),
           ("all three", diff_augment(x, ("translation", "cutout", "color"))),
           ("instance noise\n(std 0.1)", diff_augment(x, (), 0.1))]

scale, gap, top = 5, 12, 46
cell = 28 * scale
fig = Image.new("RGB", (len(columns) * (cell + gap) + gap, top + args.n * (cell + gap) + gap), "white")
draw = ImageDraw.Draw(fig)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 15)
except OSError:
    font = ImageFont.load_default()
for c, (title, imgs) in enumerate(columns):
    x0 = gap + c * (cell + gap)
    draw.multiline_text((x0 + cell // 2, 6), title, fill="black", font=font, anchor="ma", align="center")
    for r in range(args.n):
        pixels = ((imgs[r].view(28, 28).clamp(-1, 1) + 1) / 2 * 255).round().byte().numpy()
        tile = Image.fromarray(pixels, "L").resize((cell, cell), Image.NEAREST).convert("RGB")
        fig.paste(tile, (x0, top + r * (cell + gap)))
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.save(args.out)
print("saved", args.out, fig.size)
