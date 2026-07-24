# =====================================================================
#  SO-101 battery classifier — COMPLETE COLAB CELL
#
#  Open https://colab.research.google.com  -> New notebook
#  Paste this ENTIRE file into one cell and run it.
#  It will ask you to upload dataset.zip, train, then download
#  battery_cls.onnx and labels.txt automatically.
#
#  Runtime -> Change runtime type -> T4 GPU makes it ~4x faster,
#  but CPU works fine for a few hundred crops (~5 minutes).
# =====================================================================

# ---- 1. upload ------------------------------------------------------
import os
if not os.path.exists('dataset.zip') and not os.path.isdir('dataset') \
        and not os.path.isdir('ml_training/dataset'):
    from google.colab import files
    print('Choose dataset.zip ...')
    files.upload()

get_ipython().system('unzip -q -o dataset.zip')

DATA_DIR = ('ml_training/dataset' if os.path.isdir('ml_training/dataset')
            else 'dataset')
print('dataset root:', DATA_DIR)
for d in sorted(os.listdir(DATA_DIR)):
    p = os.path.join(DATA_DIR, d)
    if os.path.isdir(p):
        n = len([f for f in os.listdir(p) if f.endswith('.png')])
        flag = '' if n >= 80 else '   <-- thin'
        print(f'  {d:<18}{n:>5}{flag}')

# ---- 2. train -------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

IMG, BATCH, EPOCHS, VAL_FRAC, SEED = 224, 32, 14, 0.2, 0
torch.manual_seed(SEED)

# NO vertical flip: flipping upside-down would teach the model that upright
# and lying look alike, destroying the exact distinction being trained.
train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=12, translate=(0.08, 0.08),
                            scale=(0.85, 1.15)),
    transforms.ColorJitter(brightness=0.30, contrast=0.25,
                           saturation=0.25, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full = datasets.ImageFolder(DATA_DIR, transform=train_tf)
classes = full.classes
n = len(full)
per_class = np.bincount([y for _, y in full.samples], minlength=len(classes))
print('\nClasses:', dict(zip(classes, per_class.tolist())), '| total', n)
if n < 100:
    print('WARNING: very small dataset — expect a fragile model.')

n_val = max(1, int(n * VAL_FRAC))
train_ds, val_ds = random_split(
    full, [n - n_val, n_val], generator=torch.Generator().manual_seed(SEED))
val_ds.dataset = datasets.ImageFolder(DATA_DIR, transform=eval_tf)

# Background is usually far more numerous — weight the loss so the rarer
# battery poses are not drowned out.
weights = torch.tensor(per_class.sum() / np.maximum(per_class, 1),
                       dtype=torch.float32)
weights = weights / weights.mean()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', dev, '| train', len(train_ds), '| val', len(val_ds))

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))
model = model.to(dev)
for p in model.features.parameters():      # freeze backbone first
    p.requires_grad = False

crit = nn.CrossEntropyLoss(weight=weights.to(dev), label_smoothing=0.05)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=1e-3, weight_decay=1e-4)
tl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
vl = DataLoader(val_ds, batch_size=BATCH, num_workers=2)

best, best_state, cm = 0.0, None, np.zeros((len(classes), len(classes)), int)
for ep in range(EPOCHS):
    if ep == 4:
        for p in model.features.parameters():
            p.requires_grad = True
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        print('-- backbone unfrozen --')
    model.train()
    run = tot = cor = 0
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
    model.eval()
    cm_ep = np.zeros((len(classes), len(classes)), int)
    with torch.no_grad():
        for xb, yb in vl:
            pred = model(xb.to(dev)).argmax(1).cpu().numpy()
            for t, p in zip(yb.numpy(), pred):
                cm_ep[t, p] += 1
    va = cm_ep.trace() / max(1, cm_ep.sum())
    print(f'epoch {ep+1:2d}/{EPOCHS}  loss {run/max(1,tot):.3f}  '
          f'train {cor/max(1,tot):.3f}  val {va:.3f}')
    if va >= best:
        best, cm = va, cm_ep
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
print(f'\nBest validation accuracy: {best:.3f}')

# ---- 3. the numbers that decide whether this worked -----------------
print('\nConfusion matrix (rows = truth, cols = predicted)')
w = max(len(c) for c in classes) + 1
print(' ' * w + ''.join(f'{c[:11]:>13}' for c in classes))
for i, c in enumerate(classes):
    print(f'{c:<{w}}' + ''.join(f'{v:>13}' for v in cm[i]))

print('\nPer-class recall:')
for i, c in enumerate(classes):
    print(f'  {c:<18}{cm[i, i]/max(1, cm[i].sum()):.3f}   (n={cm[i].sum()})')

print('\nORIENTATION CONFUSION — the thing this model exists to fix:')
for a, b in (('ryobi_upright', 'ryobi_lying'), ('pink_upright', 'pink_lying')):
    if a in classes and b in classes:
        ia, ib = classes.index(a), classes.index(b)
        wrong = cm[ia, ib] + cm[ib, ia]
        tot_ab = cm[ia].sum() + cm[ib].sum()
        rate = wrong / max(1, tot_ab)
        verdict = ('SOLVED' if rate <= 0.05 else
                   'usable' if rate <= 0.12 else 'NOT GOOD ENOUGH')
        print(f'  {a} <-> {b}: {rate:.1%}  -> {verdict}')
        if rate > 0.12:
            print('     -> collect more crops of the confusing pose '
                  '(usually a lying battery seen end-on) and retrain.')

# ---- 4. export + download ------------------------------------------
torch.onnx.export(model.cpu().eval(), torch.zeros(1, 3, IMG, IMG),
                  'battery_cls.onnx', input_names=['input'],
                  output_names=['logits'],
                  dynamic_axes={'input': {0: 'n'}, 'logits': {0: 'n'}},
                  opset_version=12)
with open('labels.txt', 'w') as f:
    f.write('\n'.join(classes))
print('\nWrote battery_cls.onnx and labels.txt')

from google.colab import files as _f
_f.download('battery_cls.onnx')
_f.download('labels.txt')
print('\nPut BOTH files in SO101_ACTIVE/ next to main_ml.py, then run:')
print('  python ml_classify.py     # verify the export')
print('  python main_ml.py         # run the robot')
