import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataset_utils import SonarDataset, find_dataset_roots
from model import SonarDeconvCNN, BaselineCNN

SEED = 42
EPOCHS = 12
BASELINE_EPOCHS = 8
BATCH_SIZE = 16
IMG_SIZE = 128
LR = 1e-3
RECON_WEIGHT = 0.7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models'
RESULTS_DIR = ROOT / 'results'
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def evaluate_proposed(model, loader):
    model.eval(); ys=[]; ps=[]; losses=[]; snr_b=[]; snr_a=[]
    ce = nn.CrossEntropyLoss(); mse = nn.MSELoss()
    with torch.no_grad():
        for noisy, clean, y in loader:
            noisy, clean, y = noisy.to(DEVICE), clean.to(DEVICE), y.to(DEVICE)
            logits, recon, _ = model(noisy)
            loss = ce(logits, y) + RECON_WEIGHT*mse(recon, clean)
            losses.append(loss.item())
            pred = logits.argmax(1)
            ys.extend(y.cpu().numpy()); ps.extend(pred.cpu().numpy())
            for n, r, c in zip(noisy, recon, clean):
                n=n.squeeze(0).cpu().numpy(); r=r.squeeze(0).cpu().numpy(); c=c.squeeze(0).cpu().numpy()
                snr_b.append(10*np.log10((np.mean(c*c)+1e-8)/(np.mean((n-c)**2)+1e-8)))
                snr_a.append(10*np.log10((np.mean(c*c)+1e-8)/(np.mean((r-c)**2)+1e-8)))
    p,r,f,_=precision_recall_fscore_support(ys, ps, average='weighted', zero_division=0)
    return {
        'accuracy': float(accuracy_score(ys,ps)), 'precision': float(p), 'recall': float(r), 'f1': float(f),
        'loss': float(np.mean(losses)), 'snr_before_db': float(np.mean(snr_b)),
        'snr_after_db': float(np.mean(snr_a)), 'snr_improvement_db': float(np.mean(snr_a)-np.mean(snr_b)),
        'test_samples': len(ys), 'confusion_matrix': confusion_matrix(ys,ps,labels=[0,1]).tolist()
    }


def evaluate_baseline(model, loader):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for noisy, _, y in loader:
            logits=model(noisy.to(DEVICE)); ys.extend(y.numpy()); ps.extend(logits.argmax(1).cpu().numpy())
    p,r,f,_=precision_recall_fscore_support(ys,ps,average='weighted',zero_division=0)
    return {'accuracy':float(accuracy_score(ys,ps)), 'precision':float(p), 'recall':float(r), 'f1':float(f), 'test_samples':len(ys), 'confusion_matrix':confusion_matrix(ys,ps,labels=[0,1]).tolist()}


def main():
    seed_all()
    roots = find_dataset_roots(ROOT)
    if not roots:
        raise RuntimeError('No sonar image dataset found near the project/Downloads. Keep your extracted dataset where it already is.')
    ds = SonarDataset(roots, img_size=IMG_SIZE, noise_std=0.10)
    labels=np.array(ds.labels)
    idx=np.arange(len(ds))
    tr, te=train_test_split(idx,test_size=0.20,random_state=SEED,stratify=labels)
    tr, va=train_test_split(tr,test_size=0.20,random_state=SEED,stratify=labels[tr])
    train_loader=DataLoader(Subset(ds,tr),batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    val_loader=DataLoader(Subset(ds,va),batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
    test_loader=DataLoader(Subset(ds,te),batch_size=BATCH_SIZE,shuffle=False,num_workers=0)

    model=SonarDeconvCNN().to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    ce=nn.CrossEntropyLoss(); mse=nn.MSELoss(); history=[]; best=-1
    for ep in range(1,EPOCHS+1):
        model.train(); total=0; correct=0; n=0; running=0
        for noisy,clean,y in train_loader:
            noisy,clean,y=noisy.to(DEVICE),clean.to(DEVICE),y.to(DEVICE)
            opt.zero_grad(); logits,recon,_=model(noisy)
            loss=ce(logits,y)+RECON_WEIGHT*mse(recon,clean); loss.backward(); opt.step()
            running+=loss.item(); total+=len(y); correct+=(logits.argmax(1)==y).sum().item(); n+=1
        val=evaluate_proposed(model,val_loader)
        print(f'Epoch {ep}/{EPOCHS} | train_loss {running/max(n,1):.4f} | train_acc {correct/max(total,1):.3f} | val_acc {val["accuracy"]:.3f}')
        history.append({'epoch':ep,'train_loss':running/max(n,1),'train_accuracy':correct/max(total,1),'val_accuracy':val['accuracy'],'val_f1':val['f1']})
        if val['accuracy']>best:
            best=val['accuracy']; torch.save(model.state_dict(),MODEL_DIR/'sonar_prj44.pth')

    model.load_state_dict(torch.load(MODEL_DIR/'sonar_prj44.pth',map_location=DEVICE))
    metrics=evaluate_proposed(model,test_loader)
    metrics.update({'device':str(DEVICE),'best_val_accuracy':float(best),'model':'DeconvFeatureDenoisingCNN','reconstruction_weight':RECON_WEIGHT,'test_samples':len(te)})
    (RESULTS_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2))
    (RESULTS_DIR/'history.json').write_text(json.dumps(history,indent=2))

    # Baseline ablation: same encoder capacity, no deconvolution branch.
    base=BaselineCNN().to(DEVICE); optb=torch.optim.Adam(base.parameters(),lr=LR); cb=nn.CrossEntropyLoss()
    for ep in range(1,BASELINE_EPOCHS+1):
        base.train()
        for noisy,_,y in train_loader:
            optb.zero_grad(); logits=base(noisy.to(DEVICE)); loss=cb(logits,y.to(DEVICE)); loss.backward(); optb.step()
    bmetrics=evaluate_baseline(base,test_loader)
    (RESULTS_DIR/'baseline_metrics.json').write_text(json.dumps(bmetrics,indent=2))
    comparison={'baseline':bmetrics,'proposed':metrics,'accuracy_gain':metrics['accuracy']-bmetrics['accuracy'],'f1_gain':metrics['f1']-bmetrics['f1']}
    (RESULTS_DIR/'comparison.json').write_text(json.dumps(comparison,indent=2))
    print('')
    print('PROPOSED MODEL TEST METRICS:', json.dumps(metrics,indent=2))
    print('BASELINE METRICS:', json.dumps(bmetrics,indent=2))
    print('MODEL SAVED:', MODEL_DIR/'sonar_prj44.pth')
    print('NOVELTY ABLATION SAVED:', RESULTS_DIR/'comparison.json')

if __name__=='__main__': main()
