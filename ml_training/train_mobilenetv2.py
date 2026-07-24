"""Train a MobileNetV2 crop classifier for the SO-101 battery sorter.

Five classes: ryobi_upright, ryobi_lying, pink_upright, pink_lying, background.
One small model replaces BOTH the darkness colour heuristic AND the whole
orientation problem, and the background class kills the phantom locks.

WHERE TO RUN THIS
  Google Colab is the path of least resistance — torch is preinstalled and
  you avoid Python 3.14 wheel availability entirely. Zip the dataset, upload,
  run. A few hundred crops trains in ~2 minutes on a free GPU, and works
  fine on CPU too.

    1. zip -r dataset.zip ml_training/dataset
    2. new Colab notebook -> upload dataset.zip
    3. !unzip -q dataset.zip
    4. paste this file in a cell (or upload and %run it)
    5. download battery_cls.onnx + labels.txt when it finishes

  Locally instead: needs a Python 3.11/3.12 venv with
      pip install torch torchvision onnx
  (torch has no 3.14 wheels at time of writing).

WHAT IT PRINTS
  Per-class accuracy and a confusion matrix. The number that matters is the
  upright<->lying confusion INSIDE each colour — that is the thing five
  hand-written heuristics failed to separate. Under ~5% and it is solved.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

DATA_DIR = os.environ.get('DATA_DIR', 'ml_training/dataset')
if not os.path.isdir(DATA_DIR):          # Colab layout after unzip
    DATA_DIR = 'dataset' if os.path.isdir('dataset') else DATA_DIR
IMG = 224
BATCH = 32
EPOCHS = 14
VAL_FRAC = 0.2
OUT_ONNX = 'battery_cls.onnx'
SEED = 0

# ---------------------------------------------------------------- transforms
# NOTE: NO vertical flip. Flipping upside-down teaches the model that upright
# and lying look alike, which destroys the exact distinction we are training.
# Horizontal flip is fine — a battery is symmetric left-to-right.
train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=12, translate=(0.08, 0.08),
                            scale=(0.85, 1.15)),
    transforms.ColorJitter(brightness=0.30, contrast=0.25, saturation=0.25,
                           hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main():
    torch.manual_seed(SEED)
    if not os.path.isdir(DATA_DIR):
        sys.exit(f'No dataset at {DATA_DIR!r}. Run collect_crops.py first, '
                 'or set DATA_DIR.')

    full = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    classes = full.classes
    n = len(full)
    if n < 100:
        print(f'WARNING: only {n} images. Expect a fragile model — '
              'aim for 150+ per battery class.')
    per_class = np.bincount([y for _, y in full.samples], minlength=len(classes))
    print('Classes:', dict(zip(classes, per_class.tolist())))

    n_val = max(1, int(n * VAL_FRAC))
    train_ds, val_ds = random_split(
        full, [n - n_val, n_val],
        generator=torch.Generator().manual_seed(SEED))
    val_ds.dataset = datasets.ImageFolder(DATA_DIR, transform=eval_tf)

    # Class imbalance: background is usually far more numerous. Weight the
    # loss so the rare battery poses are not drowned out.
    weights = torch.tensor((per_class.sum() / np.maximum(per_class, 1)),
                           dtype=torch.float32)
    weights = weights / weights.mean()

    dev = ('cuda' if torch.cuda.is_available()
           else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print('Device:', dev, '| train', len(train_ds), '| val', len(val_ds))

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model = model.to(dev)

    # Transfer learning: freeze the backbone for the first few epochs so the
    # fresh head does not wreck the pretrained features, then unfreeze.
    for p in model.features.parameters():
        p.requires_grad = False

    crit = nn.CrossEntropyLoss(weight=weights.to(dev), label_smoothing=0.05)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3,
        weight_decay=1e-4)
    tl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
    vl = DataLoader(val_ds, batch_size=BATCH, num_workers=2)

    best, best_state = 0.0, None
    for ep in range(EPOCHS):
        if ep == 4:                      # unfreeze and drop the learning rate
            for p in model.features.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=2e-4,
                                    weight_decay=1e-4)
            print('-- backbone unfrozen --')
        model.train()
        tot = cor = 0
        run = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            run += loss.item() * yb.size(0)
            cor += (out.argmax(1) == yb).sum().item()
            tot += yb.size(0)
        tr_acc = cor / max(1, tot)

        model.eval()
        cm = np.zeros((len(classes), len(classes)), int)
        with torch.no_grad():
            for xb, yb in vl:
                pred = model(xb.to(dev)).argmax(1).cpu().numpy()
                for t, p in zip(yb.numpy(), pred):
                    cm[t, p] += 1
        va = cm.trace() / max(1, cm.sum())
        print(f'epoch {ep + 1:2d}/{EPOCHS}  loss {run / max(1, tot):.3f}  '
              f'train {tr_acc:.3f}  val {va:.3f}')
        if va >= best:
            best, best_state = va, {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f'\nBest validation accuracy: {best:.3f}')

    # ------------------------------------------------ the numbers that matter
    print('\nConfusion matrix (rows = truth, cols = predicted)')
    w = max(len(c) for c in classes) + 1
    print(' ' * w + ''.join(f'{c[:11]:>12}' for c in classes))
    for i, c in enumerate(classes):
        print(f'{c:<{w}}' + ''.join(f'{v:>12}' for v in cm[i]))
    print('\nPer-class recall:')
    for i, c in enumerate(classes):
        tot_i = cm[i].sum()
        print(f'  {c:<16} {cm[i, i] / max(1, tot_i):.3f}  (n={tot_i})')

    def idx(name):
        return classes.index(name) if name in classes else None
    print('\nORIENTATION CONFUSION — the thing this model exists to fix:')
    for a, b in (('ryobi_upright', 'ryobi_lying'),
                 ('pink_upright', 'pink_lying')):
        ia, ib = idx(a), idx(b)
        if ia is None or ib is None:
            continue
        wrong = cm[ia, ib] + cm[ib, ia]
        tot_ab = cm[ia].sum() + cm[ib].sum()
        rate = wrong / max(1, tot_ab)
        verdict = ('SOLVED' if rate <= 0.05 else
                   'usable' if rate <= 0.12 else 'NOT GOOD ENOUGH')
        print(f'  {a} <-> {b}: {rate:.1%}  -> {verdict}')
        if rate > 0.12:
            print('     add more crops of the confusing pose (usually a lying'
                  ' battery seen end-on) and retrain.')

    torch.onnx.export(
        model.cpu().eval(), torch.zeros(1, 3, IMG, IMG), OUT_ONNX,
        input_names=['input'], output_names=['logits'],
        dynamic_axes={'input': {0: 'n'}, 'logits': {0: 'n'}}, opset_version=12)
    with open('labels.txt', 'w') as f:
        f.write('\n'.join(classes))
    print(f'\nWrote {OUT_ONNX} and labels.txt')
    print('Normalisation used at inference: mean [0.485,0.456,0.406] '
          'std [0.229,0.224,0.225], RGB, 224x224.')


if __name__ == '__main__':
    main()
