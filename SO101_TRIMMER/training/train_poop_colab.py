# --- make the project's flat imports work from any subfolder (see paths.py) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
_R = _R if (_R / 'paths.py').is_file() else _R.parent
for _p in ('', 'core', 'calibrate', 'run', 'dashboard', 'training', 'tools'):
    _d = str(_R / _p) if _p else str(_R)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
# -----------------------------------------------------------------------------
# =====================================================================
#  POOP tile classifier — COMPLETE COLAB CELL (poop / grass / other)
#
#  1. On your Mac:  zip -r poop_dataset.zip poop_dataset
#  2. Colab -> New notebook -> paste this whole file into a cell -> run.
#     Upload poop_dataset.zip when the picker appears.
#  3. Download poop_cls.onnx + poop_labels.txt, drop them in SO101_TRIMMER/.
#     Verify the export with:  python poop_infer.py
#
#  Same recipe as the grass/battery models: MobileNetV2, backbone FROZEN (so it
#  generalises instead of memorising), strong augmentation, class-weighted,
#  single-file ONNX. With synthetic data expect a high val score that will NOT
#  fully transfer to real lawns — add real photos (fetch_poop_dataset.py /
#  collect_poop.py) and retrain to close the gap.
# =====================================================================
import os
if not os.path.exists('poop_dataset.zip') and not os.path.isdir('poop_dataset'):
    from google.colab import files
    print('Choose poop_dataset.zip ...'); files.upload()
get_ipython().system('unzip -q -o poop_dataset.zip')
DATA = 'poop_dataset' if os.path.isdir('poop_dataset') else '.'

import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

IMG, BATCH, EPOCHS, VAL, SEED, CAP = 224, 48, 20, 0.2, 0, 3.0
torch.manual_seed(SEED)

# Poop is viewed roughly top-down and has no canonical up/down, so BOTH flips
# are fine. Heavy colour/blur aug for lighting + grass-colour robustness.
train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.05),
    transforms.GaussianBlur(3, (0.1, 1.5)),
    transforms.ToTensor(), transforms.RandomErasing(0.25),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

full = datasets.ImageFolder(DATA, transform=train_tf)
classes = full.classes
per = np.bincount([y for _, y in full.samples], minlength=len(classes))
print('classes:', dict(zip(classes, per.tolist())), '| total', len(full))
nval = max(1, int(len(full) * VAL))
tr, va = random_split(full, [len(full) - nval, nval],
                      generator=torch.Generator().manual_seed(SEED))
va.dataset = datasets.ImageFolder(DATA, transform=eval_tf)
w = per.sum() / np.maximum(per, 1); w = np.minimum(w / w.min(), CAP)
weights = torch.tensor(w / w.mean(), dtype=torch.float32)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
for p in m.features.parameters():
    p.requires_grad = False
m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.last_channel, len(classes)))
m = m.to(dev)
crit = nn.CrossEntropyLoss(weight=weights.to(dev), label_smoothing=0.1)
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                        lr=1.2e-3, weight_decay=2e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
tl = DataLoader(tr, BATCH, shuffle=True, num_workers=2)
vl = DataLoader(va, BATCH, num_workers=2)

best, bstate, cm = 0., None, np.zeros((len(classes),) * 2, int)
for ep in range(EPOCHS):
    m.train(); cor = tot = 0
    for xb, yb in tl:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad(); out = m(xb); loss = crit(out, yb)
        loss.backward(); opt.step()
        cor += (out.argmax(1) == yb).sum().item(); tot += yb.size(0)
    sched.step(); m.eval(); c = np.zeros_like(cm)
    with torch.no_grad():
        for xb, yb in vl:
            pr = m(xb.to(dev)).argmax(1).cpu().numpy()
            for tr_, pr_ in zip(yb.numpy(), pr):
                c[tr_, pr_] += 1
    vacc = c.trace() / max(1, c.sum())
    print(f'epoch {ep+1:2d}/{EPOCHS}  train {cor/max(1,tot):.3f}  val {vacc:.3f}')
    if vacc >= best:
        best, cm = vacc, c
        bstate = {k: v.cpu().clone() for k, v in m.state_dict().items()}
m.load_state_dict(bstate)
print(f'\nbest val {best:.3f}')
print('confusion (rows=truth):'); print(classes); print(cm)

get_ipython().system('pip install -q onnx')
torch.onnx.export(m.cpu().eval(), torch.zeros(1, 3, IMG, IMG), 'poop_cls.onnx',
                  input_names=['input'], output_names=['logits'],
                  dynamic_axes={'input': {0: 'n'}, 'logits': {0: 'n'}},
                  opset_version=12, dynamo=False)
open('poop_labels.txt', 'w').write('\n'.join(classes))
print('size MB:', round(os.path.getsize('poop_cls.onnx') / 1e6, 2))
from google.colab import files as _f
_f.download('poop_cls.onnx'); _f.download('poop_labels.txt')
