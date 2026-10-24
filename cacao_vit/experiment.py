"""Cacao ViT protocol. Original cacao images are not included. See README.md."""
import argparse, csv, hashlib, importlib.metadata, json, math, os, platform, random, time
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, average_precision_score, log_loss)
from transformers import AutoConfig, AutoModelForImageClassification, get_linear_schedule_with_warmup

CLASSES = ['whole_beans','broken_beans','bean_fractions','skin_damaged_beans',
           'fermented_beans','unfermented_beans','moldy_beans']
COUNTS = [1187,1046,426,822,916,1776,1255]
MODELS = {
 'vit_base': ('google/vit-base-patch16-224-in21k','b4569560a39a0f1af58e3ddaf17facf20ab919b0','vit',768,12),
 'msn_small': ('facebook/vit-msn-small','a50267de5e56f5b66a7fec6d2d8a8c60db24a704','vit_msn',384,6),
 'msn_base': ('facebook/vit-msn-base','8c7214355ddee1c0e55633ceea978e75eb2b8218','vit_msn',768,12)}

def dump(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def read_csv(path):
    with open(path,newline='') as f: return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)

def seed(value):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.set_num_threads(min(4,os.cpu_count() or 1))

def worker_seed(_):
    value=torch.initial_seed()%(2**32); random.seed(value); np.random.seed(value)

def environment():
    cpu=platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                  if s.startswith('model name')),cpu)
    return {'python':platform.python_version(),'platform':platform.platform(),'cpu':cpu,
      'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
      'cuda':torch.version.cuda,'packages':{k:importlib.metadata.version(k) for k in
      ['torch','torchvision','transformers','huggingface-hub','numpy','scipy','scikit-learn','Pillow','matplotlib']}}

def model(name, pretrained=False, tiny=False):
    checkpoint,revision,kind,hidden,heads=MODELS[name]
    config=AutoConfig.for_model(kind,hidden_size=32 if tiny else hidden,
        num_attention_heads=4 if tiny else heads,num_hidden_layers=2 if tiny else 12,
        intermediate_size=64 if tiny else hidden*4,image_size=32 if tiny else 224,
        patch_size=8 if tiny else 16,num_labels=7,
        id2label=dict(enumerate(CLASSES)),label2id={v:i for i,v in enumerate(CLASSES)})
    if pretrained:
        # Keep checkpoint configuration (including MSN LayerNorm epsilon) authoritative.
        config=AutoConfig.from_pretrained(checkpoint,revision=revision,num_labels=7,
            id2label=dict(enumerate(CLASSES)),label2id={v:i for i,v in enumerate(CLASSES)})
        return AutoModelForImageClassification.from_pretrained(checkpoint,revision=revision,
            config=config,ignore_mismatched_sizes=True,attn_implementation='eager')
    return AutoModelForImageClassification.from_config(config,attn_implementation='eager')

def pixel_hash(path):
    with Image.open(path) as opened:
        im=ImageOps.exif_transpose(opened).convert('RGB')
        return hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()

def prepare(root,out,group_file=None,image_level=False,reviewed=False,allow_mismatch=False):
    root,out=Path(root).resolve(),Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    if not root.is_dir(): raise ValueError('Original image archive is required.')
    if not reviewed: raise ValueError('Review near duplicates and labels first, then pass --reviewed.')
    if not group_file and not image_level:
        raise ValueError('Supply physical bean/batch groups.csv or acknowledge weaker independence with --image-level.')
    metadata={r['path']:r['group_id'] for r in read_csv(group_file)} if group_file else None
    rows=[]
    for label,classname in enumerate(CLASSES):
        directory=root/classname
        if not directory.is_dir(): raise ValueError('Missing class folder '+classname)
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff','.bmp','.webp'}:
                relative=path.relative_to(root).as_posix(); digest=pixel_hash(path)
                if metadata is not None and not metadata.get(relative): raise ValueError('Incomplete group metadata')
                rows.append({'path':relative,'label':label,'class':classname,'sha256_pixels':digest,
                  'group_id':metadata[relative] if metadata else digest})
    counts=[sum(r['label']==i for r in rows) for i in range(7)]
    dump(out/'inventory.json',{'counts':dict(zip(CLASSES,counts)), 'source_counts_match':counts==COUNTS,
        'group_metadata_supplied':bool(metadata),'near_duplicate_review_confirmed':reviewed})
    if counts!=COUNTS and not allow_mismatch: raise ValueError('Counts differ from the source; inspect inventory.json.')
    if min(counts)<20: raise ValueError('At least 20 images per class are required.')
    # Union supplied groups connected by identical decoded pixels.
    parent={r['group_id']:r['group_id'] for r in rows}
    def find(g):
        while parent[g]!=g: parent[g]=parent[parent[g]]; g=parent[g]
        return g
    hashes={}
    for r in rows:
        if r['sha256_pixels'] in hashes:
            previous=hashes[r['sha256_pixels']]
            if previous['label']!=r['label']: raise ValueError('Identical pixels have conflicting labels.')
            parent[find(r['group_id'])]=find(previous['group_id'])
        else: hashes[r['sha256_pixels']]=r
    for r in rows: r['group_id']=find(r['group_id'])
    y=np.array([r['label'] for r in rows]); groups=np.array([r['group_id'] for r in rows])
    if any(len(set(groups[y==i]))<10 for i in range(7)): raise ValueError('Insufficient independent groups per class.')
    remaining,test=next(StratifiedGroupKFold(10,shuffle=True,random_state=42).split(y,y,groups))
    train,val=next(StratifiedGroupKFold(6,shuffle=True,random_state=42).split(y[remaining],y[remaining],groups[remaining]))
    partitions={'train':remaining[train],'validation':remaining[val],'test':test}
    for name,ids in partitions.items():
        if set(y[ids])!=set(range(7)): raise ValueError('A partition lacks a class; revise the registered split.')
        for i in ids: rows[i]['split']=name
    group_sets=[set(groups[ids]) for ids in partitions.values()]
    assert all(not group_sets[i]&group_sets[j] for i in range(3) for j in range(i+1,3))
    write_csv(out/'manifest.csv',rows,['path','label','class','sha256_pixels','group_id','split'])
    dump(out/'split_counts.json',{s:dict(Counter(CLASSES[y[i]] for i in ids)) for s,ids in partitions.items()})

class Pad:
    def __init__(self,size): self.size=size
    def __call__(self,im):
        return ImageOps.pad(im,(self.size,self.size),method=Image.Resampling.BICUBIC,color=(127,127,127))

def transform(name,training,size=224):
    ops=[T.RandomHorizontalFlip(.5),T.RandomVerticalFlip(.5),T.RandomRotation(15,expand=True,fill=(127,127,127))] if training else []
    mean,std=([.5]*3,[.5]*3) if name=='vit_base' else ([.485,.456,.406],[.229,.224,.225])
    return T.Compose(ops+[Pad(size),T.ToTensor(),T.Normalize(mean,std)])

class Images(Dataset):
    def __init__(self,root,rows,trans): self.root,self.rows,self.trans=Path(root),rows,trans
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.root/self.rows[i]['path']) as im:
            value=self.trans(ImageOps.exif_transpose(im).convert('RGB'))
        return value,int(self.rows[i]['label'])

def batches(root,rows,name,training,batch,workers,random_seed,tiny=False):
    return DataLoader(Images(root,rows,transform(name,training,32 if tiny else 224)),batch_size=batch,
        shuffle=training,num_workers=workers,worker_init_fn=worker_seed,
        generator=torch.Generator().manual_seed(random_seed),pin_memory=torch.cuda.is_available())

@torch.no_grad()
def predict(net,data,device):
    net.eval(); y=[]; p=[]
    for x,label in data:
        p.append(net(x.to(device)).logits.float().softmax(-1).cpu().numpy()); y.extend(label.tolist())
    return np.array(y),np.concatenate(p)

def metrics(y,p):
    p=np.asarray(p,dtype=float)
    if p.shape!=(len(y),7) or not np.isfinite(p).all() or (p<0).any() or (p.sum(1)<=0).any():
        raise ValueError('Invalid seven-class probabilities.')
    p=p/p.sum(1,keepdims=True); pred=p.argmax(1); hot=np.eye(7)[y]; conf=p.max(1); correct=pred==y
    ece=0.
    for b in range(15):
        selected=(conf>=b/15)&((conf<(b+1)/15) if b<14 else (conf<=1))
        if selected.any(): ece+=selected.mean()*abs(correct[selected].mean()-conf[selected].mean())
    return {'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
      'macro_f1':float(f1_score(y,pred,labels=range(7),average='macro',zero_division=0)),
      'weighted_f1':float(f1_score(y,pred,labels=range(7),average='weighted',zero_division=0)),
      'macro_ovr_roc_auc':float(roc_auc_score(y,p,labels=range(7),multi_class='ovr',average='macro')) if len(set(y))==7 else None,
      'macro_average_precision':float(average_precision_score(hot,p,average='macro')) if len(set(y))==7 else None,
      'log_loss':float(log_loss(y,p,labels=range(7))), 'brier_multiclass_sum':float(np.mean(np.sum((p-hot)**2,axis=1))),
      'ece_15_bins':float(ece),'confusion_matrix':confusion_matrix(y,pred,labels=range(7)).tolist(),
      'class_report':classification_report(y,pred,labels=range(7),target_names=CLASSES,zero_division=0,output_dict=True)}

def verify_images(root,rows):
    for r in rows:
        if pixel_hash(Path(root)/r['path'])!=r['sha256_pixels']: raise ValueError('Image changed after partitioning: '+r['path'])

def train(args,tiny=False):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False); seed(args.seed)
    rows=read_csv(args.manifest); tr=[r for r in rows if r['split']=='train']; va=[r for r in rows if r['split']=='validation']
    verify_images(args.root,tr+va)
    net=model(args.model,pretrained=not tiny,tiny=tiny); device='cuda' if torch.cuda.is_available() else 'cpu'; net.to(device)
    if args.freeze:
        for key,param in net.named_parameters(): param.requires_grad=key.startswith('classifier.')
    training=batches(args.root,tr,args.model,True,args.batch,args.workers,args.seed,tiny)
    validation=batches(args.root,va,args.model,False,args.batch,args.workers,args.seed,tiny)
    counts=np.bincount([int(r['label']) for r in tr],minlength=7)
    if (counts==0).any(): raise ValueError('Missing training class')
    weights=torch.tensor(len(tr)/(7*counts),dtype=torch.float32,device=device) if not args.unweighted else None
    lossfn=torch.nn.CrossEntropyLoss(weight=weights)
    optimizer=torch.optim.AdamW([{'params':[p for p in net.parameters() if p.requires_grad and p.ndim>1],'weight_decay':.02},
      {'params':[p for p in net.parameters() if p.requires_grad and p.ndim<=1],'weight_decay':0}],lr=1e-5,betas=(.9,.999),eps=1e-8)
    steps=math.ceil(len(training)/args.accumulation)*args.epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,math.ceil(.1*steps),steps)
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda')
    settings=vars(args).copy(); settings.update(environment=environment(),synthetic_software_check_only=tiny,
      checkpoint=MODELS[args.model][0],revision=MODELS[args.model][1],manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest())
    dump(out/'configuration.json',settings); history=[]; best=-1.; stale=0; started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        net.train()
        if args.freeze: net.base_model.eval()
        optimizer.zero_grad(set_to_none=True); losses=[]
        for i,(x,y) in enumerate(training):
            length=min(args.accumulation,len(training)-(i//args.accumulation)*args.accumulation)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
                raw=lossfn(net(x.to(device)).logits,y.to(device)); loss=raw/length
            if not torch.isfinite(loss): raise ValueError('Non-finite loss; run incomplete')
            scaler.scale(loss).backward(); losses.append(float(raw.detach()))
            if (i+1)%args.accumulation==0 or i+1==len(training):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                before=scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale()>=before: scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        y,p=predict(net,validation,device); performance=metrics(y,p)
        history.append({'epoch':epoch,'training_loss_batch_mean':float(np.mean(losses)),'validation':performance})
        dump(out/'history.json',history); print('Epoch',epoch,'validation macro F1',performance['macro_f1'],flush=True)
        if performance['macro_f1']>best:
            best=performance['macro_f1']; stale=0; net.save_pretrained(out/'best_model')
            dump(out/'selection.json',{'epoch':epoch,'validation_macro_f1':best})
        else: stale+=1
        if stale>=args.patience: break
    dump(out/'training_completed.json',{'completed':True,'seconds':time.perf_counter()-started,
      'synthetic_software_check_only':tiny,'cacao_training_executed':not tiny})

def evaluate(args):
    run=Path(args.run); config=json.loads((run/'configuration.json').read_text())
    if not (run/'training_completed.json').exists(): raise ValueError('Training is incomplete')
    if (run/'test_metrics.json').exists(): raise ValueError('Test evaluation exists; refusing silent replacement')
    if config['manifest_sha256']!=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(): raise ValueError('Different manifest')
    rows=[r for r in read_csv(args.manifest) if r['split']=='test']; verify_images(args.root,rows)
    seed(config['seed']); device='cuda' if torch.cuda.is_available() else 'cpu'
    net=AutoModelForImageClassification.from_pretrained(run/'best_model',attn_implementation='eager').to(device)
    data=batches(args.root,rows,config['model'],False,config['batch'],0,config['seed'],config['synthetic_software_check_only'])
    y,p=predict(net,data,device); report=metrics(y,p)
    groups=np.array([r['group_id'] for r in rows]); ids=sorted(set(groups)); members={g:np.flatnonzero(groups==g) for g in ids}
    rng=np.random.default_rng(2026); samples=[]
    for _ in range(args.bootstrap):
        sample=np.concatenate([members[g] for g in rng.choice(ids,len(ids),replace=True)])
        samples.append([accuracy_score(y[sample],p[sample].argmax(1)),f1_score(y[sample],p[sample].argmax(1),labels=range(7),average='macro',zero_division=0)])
    report['group_bootstrap_95_ci']=np.quantile(samples,[.025,.975],axis=0).tolist()
    report['bootstrap_columns']=['accuracy','macro_f1']; report['bootstrap_repetitions']=args.bootstrap
    report['synthetic_software_check_only']=config['synthetic_software_check_only']; dump(run/'test_metrics.json',report)
    predictions=[dict(path=r['path'],group_id=r['group_id'],label=int(r['label']),prediction=int(v.argmax()),
                  **{f'p_{i}':float(v[i]) for i in range(7)}) for r,v in zip(rows,p)]
    write_csv(run/'test_predictions.csv',predictions,['path','group_id','label','prediction']+[f'p_{i}' for i in range(7)])
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve,precision_recall_curve
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for i,label in enumerate(CLASSES):
        fpr,tpr,_=roc_curve(y==i,p[:,i]); precision,recall,_=precision_recall_curve(y==i,p[:,i])
        axes[0].plot(fpr,tpr); axes[1].plot(recall,precision,label=label)
    axes[0].set(xlabel='False positive rate',ylabel='True positive rate')
    axes[1].set(xlabel='Recall',ylabel='Precision'); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(run/'roc_pr.png',dpi=250); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6)); cm=np.array(report['confusion_matrix']); ax.imshow(cm,cmap='Blues')
    for i in range(7):
        for j in range(7): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    ax.set_xticks(range(7),CLASSES,rotation=50,ha='right'); ax.set_yticks(range(7),CLASSES)
    ax.set(xlabel='Predicted class',ylabel='True class'); fig.tight_layout(); fig.savefig(run/'confusion_matrix.png',dpi=250); plt.close(fig)
    # Exploratory CLS attention rollout, fixed first test image rather than selected examples.
    x,_=next(iter(data)); x=x[:1].to(device); net.eval()
    with torch.no_grad(): att=net(x,output_attentions=True).attentions
    joint=torch.eye(att[0].shape[-1],device=device)
    for a in att:
        mat=a[0].mean(0)+torch.eye(a.shape[-1],device=device); joint=(mat/mat.sum(-1,keepdim=True))@joint
    side=x.shape[-1]//net.config.patch_size
    fig,ax=plt.subplots(figsize=(4,4)); ax.imshow(joint[0,1:].reshape(side,side).cpu(),cmap='magma'); ax.axis('off')
    fig.savefig(run/'attention_rollout.png',dpi=250,bbox_inches='tight'); plt.close(fig)

def audit(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=False); seed(42); records=[]
    for name in MODELS:
        net=model(name); net.eval()
        with torch.no_grad(): shape=list(net(torch.zeros(1,3,224,224)).logits.shape)
        assert shape==[1,7]
        records.append({'model':name,'parameters':sum(p.numel() for p in net.parameters()),'shape':shape,
          'checkpoint':MODELS[name][0],'revision':MODELS[name][1],'random_initialization':True,'cacao_training_executed':False})
        del net
    dump(out/'model_audit.json',{'models':records,'environment':environment()}); print(records)

def smoke(out):
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False); root=out/'synthetic'; rng=np.random.default_rng(42)
    for classname in CLASSES:
        directory=root/classname; directory.mkdir(parents=True)
        for i in range(100): Image.fromarray(rng.integers(0,256,(16,16,3),dtype=np.uint8)).save(directory/f'{i}.png')
    prepare(root,out/'split',image_level=True,reviewed=True,allow_mismatch=True)
    a=argparse.Namespace(root=str(root),manifest=str(out/'split/manifest.csv'),out=str(out/'tiny'),model='vit_base',
        seed=42,epochs=2,batch=8,workers=0,accumulation=2,patience=2,freeze=False,unweighted=False)
    train(a,tiny=True); evaluate(argparse.Namespace(root=str(root),manifest=a.manifest,run=a.out,bootstrap=25))
    y=np.tile(np.arange(7),3); perfect=metrics(y,np.eye(7)[y]); uniform=metrics(y,np.full((len(y),7),1/7))
    assert perfect['macro_f1']==1 and perfect['ece_15_bins']==0
    assert abs(uniform['log_loss']-math.log(7))<1e-10
    assert abs(uniform['brier_multiclass_sum']-6/7)<1e-10
    msn=model('msn_small',tiny=True)
    with torch.no_grad(): assert msn(torch.zeros(2,3,32,32)).logits.shape==(2,7)
    dump(out/'verification.json',{'status':'passed','cacao_images_used':0,'cacao_training_executed':False,
      'checks':['group separation','training and checkpoint reload','seven-class ViT and MSN outputs',
       'perfect and uniform metric checks','bootstrap','plots','attention rollout'],'environment':environment()})
    print('PASS: synthetic software checks only, no cacao results.')

def main():
    p=argparse.ArgumentParser(description=__doc__); subs=p.add_subparsers(dest='command',required=True)
    for name in ['audit','smoke']:
        s=subs.add_parser(name); s.add_argument('--out',required=True)
    s=subs.add_parser('prepare'); s.add_argument('--root',required=True); s.add_argument('--out',required=True)
    s.add_argument('--groups'); s.add_argument('--image-level',action='store_true'); s.add_argument('--reviewed',action='store_true')
    s.add_argument('--allow-count-mismatch',action='store_true')
    s=subs.add_parser('train')
    for name in ['root','manifest','out']: s.add_argument('--'+name,required=True)
    s.add_argument('--model',choices=MODELS,required=True)
    for name,value in [('seed',42),('epochs',25),('batch',8),('workers',2),('accumulation',4),('patience',5)]:
        s.add_argument('--'+name,type=int,default=value)
    s.add_argument('--freeze',action='store_true'); s.add_argument('--unweighted',action='store_true')
    s=subs.add_parser('evaluate')
    for name in ['root','manifest','run']: s.add_argument('--'+name,required=True)
    s.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args()
    if a.command=='prepare': prepare(a.root,a.out,a.groups,a.image_level,a.reviewed,a.allow_count_mismatch)
    elif a.command=='train':
        if min(a.epochs,a.batch,a.accumulation,a.patience)<1 or a.workers<0: p.error('Invalid training settings')
        train(a)
    elif a.command=='evaluate':
        if a.bootstrap<1: p.error('Bootstrap repetitions must be positive')
        evaluate(a)
    elif a.command=='audit': audit(a.out)
    else: smoke(a.out)

if __name__=='__main__': main()
