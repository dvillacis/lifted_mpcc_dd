"""Prepare the training sets of ``cpp/tv_learn`` from the downloaded archives.

Each set becomes ``images/<name>/`` with grayscale P×P PNGs in ``train/`` and
``val/`` and the two lists ``train.txt`` / ``val.txt`` that ``tv_learn --train``
/ ``--val`` read (paths relative to the list).  The PNGs are ignored by git;
the lists are tracked, so a run names exactly the images it used.  Everything
is deterministic for a given ``--seed``.

  bsds      BSDS500 (images/BSR_bsds500.tgz).  Training pool: its train (200)
            and test (200) folders, the "BSD400" of the denoising literature;
            held out: its val folder (100 images, which contains the 68 of
            BSD68).  --per-image random P×P crops of each grayscale image.
  cartoon   Google Cartoon Set 10k (images/cartoonset10k.tgz, CC BY 4.0).
            Composited on white, grayscale, area-downsampled from 500×500 to
            P×P; the files sorted by name, the first --train-count train and
            the next --val-count are held out.

  uv run python make_datasets.py bsds --size 64 --per-image 1
  uv run python make_datasets.py cartoon --size 64 --train-count 1000 --val-count 500
  uv run python make_datasets.py phantom --size 64

The archives are unpacked into images/raw/ (ignored by git) on first use.
"""

import argparse
import glob
import os
import tarfile

import numpy as np
from PIL import Image

IMAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
RAW = os.path.join(IMAGES, "raw")


def unpack(archive, marker):
    """Unpack images/<archive> into images/raw/ unless <marker> exists there."""
    if os.path.exists(os.path.join(RAW, marker)):
        return
    path = os.path.join(IMAGES, archive)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}: download it first (see the docstring)")
    os.makedirs(RAW, exist_ok=True)
    print(f"unpacking {archive} ...")
    with tarfile.open(path) as tf:
        tf.extractall(RAW, filter="data")


def write_set(name, items, size):
    """items: (role, stem, PIL grayscale image of size×size).  Writes the PNGs
    and the lists, replacing any previous version of the set."""
    root = os.path.join(IMAGES, name)
    for role in ("train", "val"):
        os.makedirs(os.path.join(root, role), exist_ok=True)
        for old in glob.glob(os.path.join(root, role, "*.png")):
            os.remove(old)
    lists = {"train": [], "val": []}
    for role, stem, img in items:
        assert img.size == (size, size) and img.mode == "L"
        rel = f"{role}/{stem}.png"
        img.save(os.path.join(root, rel))
        lists[role].append(rel)
    for role, rels in lists.items():
        with open(os.path.join(root, f"{role}.txt"), "w") as fh:
            fh.write(f"# {name}: {len(rels)} {role} images, {size}x{size} grayscale "
                     f"(python/make_datasets.py)\n")
            fh.write("\n".join(rels) + "\n")
        print(f"  {root}/{role}.txt: {len(rels)} images")


def bsds(args):
    unpack("BSR_bsds500.tgz", "BSR")
    base = os.path.join(RAW, "BSR", "BSDS500", "data", "images")
    pools = {"train": sorted(glob.glob(os.path.join(base, "train", "*.jpg")))
                      + sorted(glob.glob(os.path.join(base, "test", "*.jpg"))),
             "val": sorted(glob.glob(os.path.join(base, "val", "*.jpg")))}
    rng = np.random.default_rng(args.seed)
    P, items = args.size, []
    for role, files in pools.items():
        for f in files:
            img = Image.open(f).convert("L")
            w, h = img.size
            stem = os.path.splitext(os.path.basename(f))[0]
            for k in range(args.per_image):
                x, y = int(rng.integers(0, w - P + 1)), int(rng.integers(0, h - P + 1))
                items.append((role, f"{stem}_{k}", img.crop((x, y, x + P, y + P))))
    write_set(args.name or f"bsds{P}", items, P)


def cartoon(args):
    unpack("cartoonset10k.tgz", "cartoonset10k")
    files = sorted(glob.glob(os.path.join(RAW, "cartoonset10k", "*.png")))
    n_tr, n_va = args.train_count, args.val_count
    if n_tr + n_va > len(files):
        raise SystemExit(f"only {len(files)} cartoons")
    P, items = args.size, []
    for i, f in enumerate(files[: n_tr + n_va]):
        img = Image.open(f)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            white = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(white, img)
        img = img.convert("L").resize((P, P), Image.Resampling.BOX)
        stem = os.path.splitext(os.path.basename(f))[0]
        items.append(("train" if i < n_tr else "val", stem, img))
    write_set(args.name or f"cartoon{P}", items, P)


def phantom(args):
    P = args.size
    y, x = (np.mgrid[0:P, 0:P] + 0.5) / P
    u = np.full((P, P), 0.2)
    u[(x > 0.08) & (x < 0.55) & (y > 0.10) & (y < 0.45)] = 0.75   # rectangle
    u[(x > 0.20) & (x < 0.40) & (y > 0.20) & (y < 0.35)] = 0.45   # its inset
    u[(x - 0.72) ** 2 + (y - 0.70) ** 2 < 0.17 ** 2] = 0.9         # disk
    for i in range(3):                                            # small squares
        for j in range(3):
            if (i + j) % 2 == 0:
                x0, y0 = 0.10 + 0.11 * i, 0.60 + 0.11 * j
                u[(x >= x0) & (x < x0 + 0.08) & (y >= y0) & (y < y0 + 0.08)] = 0.6
    path = os.path.join(IMAGES, args.name or f"phantom{P}.png")
    Image.fromarray(np.round(255 * u).astype(np.uint8), mode="L").save(path)
    print(f"  {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set", choices=["bsds", "cartoon", "phantom"])
    ap.add_argument("--size", type=int, default=64, help="P: images are P×P")
    ap.add_argument("--per-image", type=int, default=1, help="bsds: crops per image")
    ap.add_argument("--train-count", type=int, default=1000, help="cartoon: training images")
    ap.add_argument("--val-count", type=int, default=500, help="cartoon: held-out images")
    ap.add_argument("--seed", type=int, default=0, help="bsds: crop positions")
    ap.add_argument("--name", default="", help="images/<name> (default bsds<P> / cartoon<P> / phantom<P>.png)")
    args = ap.parse_args()
    {"bsds": bsds, "cartoon": cartoon, "phantom": phantom}[args.set](args)


if __name__ == "__main__":
    main()
