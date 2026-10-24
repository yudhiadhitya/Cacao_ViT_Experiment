"""Actual frozen MSN-Small ViT training. Extracted representations are cached.

Requires a verified manifest from experiment.py prepare. This is a linear probe,
not full fine-tuning. All model selection uses validation macro F1.
"""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from experiment import (read_csv,model,batches,seed,environment,CLASSES,MODELS,metrics,dump,write_csv,verify_images)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
 p.add_argument('--epochs',type=int,default=100);p.add_argument('--patience',type=int,default=10)
 args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True);seed(42)
 rows=read_csv(args.manifest); device='cuda' if torch.cuda.is_available() else 'cpu'
 verify_images(args.root,rows)
 manifest_hash=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
 if not (out/'features.npz').exists():
  net=model('msn_small',pretrained=True).to(device);net.eval()
  for param in net.parameters():param.requires_grad=False
  # Deterministic preprocessing for a frozen representation baseline; no augmented views.
  # Single-process loading works in runtimes that disallow multiprocessing sockets.
  data=batches(args.root,rows,'msn_small',False,16,0,42)
  features=[];labels=[];started=time.perf_counter();count=0
  with torch.no_grad():
   for x,y in data:
    value=net.base_model(pixel_values=x.to(device)).last_hidden_state[:,0,:]
    features.append(value.cpu().numpy());labels.extend(y.tolist());count+=len(y)
    if count%512==0 or count==len(rows):print('Extracted',count,'of',len(rows),'seconds',round(time.perf_counter()-started,1),flush=True)
  features=np.concatenate(features).astype('float32');labels=np.array(labels,dtype='int64')
  np.savez_compressed(out/'features.npz',features=features,labels=labels)
  dump(out/'feature_configuration.json',{'model':'facebook/vit-msn-small','revision':MODELS['msn_small'][1],
   'manifest_sha256':manifest_hash,'samples':len(rows),'dimensions':features.shape[1],
   'extraction_seconds':time.perf_counter()-started,'backbone_frozen':True,'augmentation':False,
   'extraction_batch_size':16,'data_loading_workers':0,'environment':environment()})
  del net
 else:
  configuration=json.loads((out/'feature_configuration.json').read_text())
  if configuration['manifest_sha256']!=manifest_hash:raise ValueError('Cached feature manifest differs')
  z=np.load(out/'features.npz');features=z['features'];labels=z['labels']
 split=np.array([r['split'] for r in rows]);tr=np.flatnonzero(split=='train');va=np.flatnonzero(split=='validation');te=np.flatnonzero(split=='test')
 x=torch.tensor(features);y=torch.tensor(labels);summaries=[]
 # All six runs are prespecified before test evaluation; no test-based hyperparameter tuning.
 for weighted in [True,False]:
  for random_seed in [42,52,62]:
   seed(random_seed);tag=('weighted' if weighted else 'unweighted')+'_'+str(random_seed);run=out/tag
   if (run/'test_metrics.json').exists():summaries.append(json.loads((run/'summary.json').read_text()));continue
   run.mkdir(exist_ok=True);head=torch.nn.Linear(features.shape[1],7).to(device)
   counts=np.bincount(labels[tr],minlength=7)
   weights=torch.tensor(len(tr)/(7*counts),dtype=torch.float32,device=device) if weighted else None
   lossfn=torch.nn.CrossEntropyLoss(weight=weights)
   optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
   data=DataLoader(TensorDataset(x[tr],y[tr]),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(random_seed))
   history=[];best=-1.;stale=0;started=time.perf_counter()
   for epoch in range(1,args.epochs+1):
    head.train();losses=[]
    for xb,yb in data:
     optimizer.zero_grad();loss=lossfn(head(xb.to(device)),yb.to(device));loss.backward();optimizer.step();losses.append(float(loss.detach()))
    head.eval()
    with torch.no_grad():prob=head(x[va].to(device)).softmax(-1).cpu().numpy()
    validation=metrics(labels[va],prob)
    history.append({'epoch':epoch,'training_loss_batch_mean':float(np.mean(losses)),'validation_macro_f1':validation['macro_f1'],'validation_accuracy':validation['accuracy']})
    if validation['macro_f1']>best:
     best=validation['macro_f1'];stale=0;torch.save(head.state_dict(),run/'classifier.pt');chosen=epoch
    else:stale+=1
    if stale>=args.patience:break
   dump(run/'history.json',history)
   training_seconds=time.perf_counter()-started
   head.load_state_dict(torch.load(run/'classifier.pt',map_location=device,weights_only=True));head.eval()
   with torch.no_grad():prob=head(x[te].to(device)).softmax(-1).cpu().numpy()
   result=metrics(labels[te],prob)
   groups=np.array([rows[i]['group_id'] for i in te]);ids=sorted(set(groups));members={g:np.flatnonzero(groups==g) for g in ids}
   rng=np.random.default_rng(2026);sampled=[]
   from sklearn.metrics import accuracy_score,f1_score
   for _ in range(2000):
    sample=np.concatenate([members[g] for g in rng.choice(ids,len(ids),replace=True)])
    sampled.append([accuracy_score(labels[te][sample],prob[sample].argmax(1)),f1_score(labels[te][sample],prob[sample].argmax(1),labels=range(7),average='macro',zero_division=0)])
   result['group_bootstrap_95_ci']=np.quantile(sampled,[.025,.975],axis=0).tolist()
   result['bootstrap_columns']=['accuracy','macro_f1'];result['bootstrap_repetitions']=2000
   dump(run/'test_metrics.json',result)
   predicted=[dict(path=rows[i]['path'],group_id=rows[i]['group_id'],label=int(labels[i]),prediction=int(v.argmax()),**{f'p_{j}':float(v[j]) for j in range(7)}) for i,v in zip(te,prob)]
   write_csv(run/'test_predictions.csv',predicted,['path','group_id','label','prediction']+[f'p_{j}' for j in range(7)])
   summary={'run':tag,'seed':random_seed,'weighted':weighted,'selected_epoch':chosen,'epochs_completed':len(history),
    'best_validation_macro_f1':best,'test_accuracy':result['accuracy'],'test_macro_f1':result['macro_f1'],
    'test_balanced_accuracy':result['balanced_accuracy'],'test_macro_ovr_roc_auc':result['macro_ovr_roc_auc'],
    'test_macro_average_precision':result['macro_average_precision'],'test_log_loss':result['log_loss'],
    'test_ece':result['ece_15_bins'],'training_seconds':training_seconds,
    'run_seconds_including_evaluation':time.perf_counter()-started,
    'trainable_parameters':sum(p.numel() for p in head.parameters()),'backbone_frozen':True,
    'manifest_sha256':manifest_hash,'cacao_training_executed':True,'learning_rate':.001,'weight_decay':.0001,
    'batch_size':64,'max_epochs':args.epochs,'patience':args.patience,'device':device}
   dump(run/'summary.json',summary);summaries.append(summary);print('Completed',tag,'accuracy',result['accuracy'],'macro F1',result['macro_f1'],flush=True)
 dump(out/'all_run_summaries.json',summaries)
 aggregate={}
 for weighted in [True,False]:
  subset=[r for r in summaries if r['weighted']==weighted]
  aggregate['weighted' if weighted else 'unweighted']={metric:{'mean':float(np.mean([r[metric] for r in subset])),
   'sample_sd':float(np.std([r[metric] for r in subset],ddof=1))} for metric in ['test_accuracy','test_macro_f1','test_balanced_accuracy','test_macro_ovr_roc_auc','test_macro_average_precision']}
 dump(out/'aggregate_results.json',aggregate);print('ACTUAL EXPERIMENT COMPLETE',json.dumps(aggregate),flush=True)

if __name__=='__main__':main()
