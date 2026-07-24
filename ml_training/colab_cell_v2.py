# =====================================================================
#  SO-101 battery classifier — TRAINING v2  (generalises, doesn't memorise)
#
#  Paste this whole file into a Colab cell and run. Same upload flow as
#  before (dataset.zip + this file, or dataset.zip + colab_cell_v2.py).
#
#  WHY v2 EXISTS
#  v1 hit 100% validation the instant the backbone unfroze — that was
#  MEMORISATION, not learning. Live batteries (a slightly different
#  distribution) got dumped into 'background' at 0.9 confidence. v2 fixes
#  generalisation:
#    * backbone FROZEN the whole time (ImageNet features stay generic;
#      the model cannot memorise 400 crops)
#    * class weights CAPPED (v1 up-weighted the 30-sample background class
#      ~13x, which taught the model to over-predict background)
#    * stronger augmentation (brightness/blur/erasing) to bridge the gap to
#      live lighting
#    * more dropout
#  Expect val accuracy around 0.90-0.97 here — LOWER than v1's fake 1.000,
#  but it will actually work on live crops.
# =====================================================================
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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

IMG, BATCH, EPOCHS, VAL_FRAC, SEED = 224, 32, 22, 0.2, 0
WEIGHT_CAP = 3.0                 # was effectively ~13 for background in v1
torch.manual_seed(SEED)

# Stronger, lighting-focused augmentation. STILL no vertical flip — that
# would teach the model upright and lying look alike.
train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.10, 0.10),
                            scale=(0.80, 1.20)),
    transforms.ColorJitter(brightness=0.45, contrast=0.35,
                           saturation=0.35, hue=0.04),
    transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
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
print('Classes:', dict(zip(classes, per_class.tolist())), '| total', n)

n_val = max(1, int(n * VAL_FRAC))
train_ds, val_ds = random_split(
    full, [n - n_val, n_val], generator=torch.Generator().manual_seed(SEED))
val_ds.dataset = datasets.ImageFolder(DATA_DIR, transform=eval_tf)

# CAPPED weights: rare classes get a little boost, not a huge one, so
# background (30 samples) stops dominating the predictions.
w = per_class.sum() / np.maximum(per_class, 1)
w = np.minimum(w / w.min(), WEIGHT_CAP)
weights = torch.tensor(w / w.mean(), dtype=torch.float32)
print('class weights:', dict(zip(classes, [round(float(x), 2) for x in weights])))

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
# FREEZE the backbone — the whole point of v2.
for p in model.features.parameters():
    p.requires_grad = False
# More dropout in the head for regularisation.
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.last_channel, len(classes)))
model = model.to(dev)

crit = nn.CrossEntropyLoss(weight=weights.to(dev), label_smoothing=0.10)
opt = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1.2e-3,
    weight_decay=2e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
tl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
vl = DataLoader(val_ds, batch_size=BATCH, num_workers=2)

best, best_state, cm = 0.0, None, np.zeros((len(classes), len(classes)), int)
for ep in range(EPOCHS):
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
    sched.step()
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
print(f'\nBest validation accuracy: {best:.3f}  '
      '(0.90-0.97 is HEALTHY here; 1.000 would mean it memorised again)')

print('\nConfusion matrix (rows = truth, cols = predicted)')
w2 = max(len(c) for c in classes) + 1
print(' ' * w2 + ''.join(f'{c[:11]:>13}' for c in classes))
for i, c in enumerate(classes):
    print(f'{c:<{w2}}' + ''.join(f'{v:>13}' for v in cm[i]))
print('\nORIENTATION CONFUSION:')
for a, b in (('ryobi_upright', 'ryobi_lying'), ('pink_upright', 'pink_lying')):
    if a in classes and b in classes:
        ia, ib = classes.index(a), classes.index(b)
        wrong = cm[ia, ib] + cm[ib, ia]
        tot_ab = cm[ia].sum() + cm[ib].sum()
        print(f'  {a} <-> {b}: {wrong/max(1,tot_ab):.1%}')

get_ipython().system('pip install -q onnx')   # needed by the ONNX exporter
torch.onnx.export(model.cpu().eval(), torch.zeros(1, 3, IMG, IMG),
                  'battery_cls.onnx', input_names=['input'],
                  output_names=['logits'],
                  dynamic_axes={'input': {0: 'n'}, 'logits': {0: 'n'}},
                  opset_version=12, dynamo=False)
with open('labels.txt', 'w') as f:
    f.write('\n'.join(classes))
print('\nsize MB:', round(os.path.getsize('battery_cls.onnx') / 1e6, 2),
      '(should be ~9-14)')
from google.colab import files as _f
_f.download('battery_cls.onnx')
_f.download('labels.txt')
