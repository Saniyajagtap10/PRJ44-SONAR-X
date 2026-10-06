from pathlib import Path
import os, re
from collections import Counter

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def _scan_dirs(base, max_depth=4):
    """Permission-safe directory scan; never walks protected Windows folders."""
    base = Path(base).resolve()
    if not base.exists():
        return []
    results = []
    base_depth = len(base.parts)
    skip_names = {
        "Application Data", "AppData", "Local Settings",
        "$Recycle.Bin", "System Volume Information"
    }
    for current, dirs, files in os.walk(base, topdown=True, onerror=lambda e: None):
        # prune protected / irrelevant directories
        dirs[:] = [d for d in dirs if d not in skip_names and not d.startswith(".")]
        depth = len(Path(current).parts) - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        results.append(Path(current))
    return results

def find_dataset_roots(start=None, max_depth=4):
    """Find folders containing many matching JPG/TXT annotation pairs."""
    start = Path(start or Path.cwd()).resolve()
    candidates = [start]
    # Search the project parent, but NOT the whole user profile.
    if start.parent.exists():
        candidates.append(start.parent)
    downloads = Path.home() / "Downloads"
    if downloads.exists():
        candidates.append(downloads)

    roots, seen = [], set()
    for base in candidates:
        for p in _scan_dirs(base, max_depth=max_depth):
            key = str(p).lower()
            if key in seen:
                continue
            try:
                files = list(p.iterdir())
            except (PermissionError, OSError):
                continue
            imgs = [x for x in files if x.is_file() and x.suffix.lower() in IMAGE_EXTS]
            txts = {x.stem.lower() for x in files if x.is_file() and x.suffix.lower()==".txt"}
            pairs = [x for x in imgs if x.stem.lower() in txts]
            if len(pairs) >= 10:
                roots.append(p); seen.add(key)
    return sorted(roots, key=lambda x: (len(x.parts), -len(list(x.glob("*.jpg")))))

def find_names_file(roots):
    for r in roots:
        for base in [r, *r.parents]:
            try:
                p = base / "obj.names"
                if p.exists():
                    return p
            except OSError:
                pass
    # Only search the project tree, not the whole profile.
    for p in Path.cwd().rglob("obj.names"):
        return p
    return None

def load_class_names(names_path):
    if not names_path or not names_path.exists():
        return {0:"MILCO", 1:"NOMBO"}
    lines=[x.strip() for x in names_path.read_text(errors="ignore").splitlines() if x.strip()]
    names={i:n for i,n in enumerate(lines)}
    out={}
    for i,n in names.items():
        u=n.upper()
        if "MILCO" in u or "MINE" in u: out[i]="MILCO"
        elif "NOMBO" in u or "BOTTOM" in u or "OBJECT" in u: out[i]="NOMBO"
        else: out[i]=n
    return out or {0:"MILCO",1:"NOMBO"}

def parse_label(txt_path, class_names):
    ids=[]
    try:
        for line in txt_path.read_text(errors="ignore").splitlines():
            parts=line.strip().split()
            if not parts: continue
            if re.match(r"^-?\d+$", parts[0]):
                ids.append(int(parts[0]))
            else:
                u=line.upper()
                for k,v in class_names.items():
                    if str(v).upper() in u: ids.append(k)
    except (OSError, UnicodeError):
        pass
    if not ids: return None
    normalized=[class_names.get(i,str(i)).upper() for i in ids]
    for i,x in zip(ids,normalized):
        if "MILCO" in x or "MINE" in x: return i
    return Counter(ids).most_common(1)[0][0]

def collect_samples(roots):
    names_file=find_names_file(roots)
    names=load_class_names(names_file)
    samples=[]
    for r in roots:
        try: files=list(r.iterdir())
        except (PermissionError,OSError): continue
        for img in files:
            if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS: continue
            txt=img.with_suffix(".txt")
            if not txt.exists(): continue
            raw=parse_label(txt,names)
            if raw is None: continue
            label_name=names.get(raw,str(raw)); u=label_name.upper()
            if "MILCO" in u or "MINE" in u: label=0; label_name="MILCO"
            elif "NOMBO" in u or "BOTTOM" in u or "OBJECT" in u: label=1; label_name="NOMBO"
            elif raw in (0,1): label=raw; label_name="MILCO" if raw==0 else "NOMBO"
            else: continue
            samples.append({"image":str(img),"txt":str(txt),"label":label,"class":label_name})
    return list({x["image"]:x for x in samples}.values()), names_file, names

def summarize(samples):
    c=Counter(x["class"] for x in samples)
    return {"total":len(samples),"MILCO":c.get("MILCO",0),"NOMBO":c.get("NOMBO",0)}

# PyTorch dataset used by train.py
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class SonarDataset(Dataset):
    def __init__(self, roots, img_size=128, noise_std=0.10):
        self.img_size = int(img_size)
        self.noise_std = float(noise_std)
        samples, _, _ = collect_samples(roots)
        if not samples:
            raise RuntimeError('No labeled sonar JPG/TXT pairs were found in the detected dataset folders.')
        self.samples = samples
        self.labels = [int(s['label']) for s in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        with Image.open(s['image']) as im:
            im = im.convert('L').resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        clean = torch.from_numpy(arr).unsqueeze(0)
        noise = torch.randn_like(clean) * self.noise_std
        noisy = torch.clamp(clean + noise, 0.0, 1.0)
        label = torch.tensor(int(s['label']), dtype=torch.long)
        return noisy, clean, label
