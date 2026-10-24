from pathlib import Path
import json,csv,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,f1_score,roc_curve,precision_recall_curve
sys.path.insert(0,'cacao_vit')
from experiment import metrics,CLASSES
base=Path('actual_frozen_msn_small');out=base/'analysis';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
def read(tag):
 with open(base/tag/'test_predictions.csv') as f:r=list(csv.DictReader(f))
 return r,np.array([int(v['label']) for v in r]),np.array([[float(v[f'p_{i}']) for i in range(7)] for v in r])
rows,y,p=read('weighted_42');r2,y2,p2=read('unweighted_42')
assert [r['path'] for r in rows]==[r['path'] for r in r2] and np.array_equal(y,y2)
report=metrics(y,p);cm=np.array(report['confusion_matrix']);short=['Whole','Broken','Fraction','Skin damage','Fermented','Unfermented','Moldy']
fig,axes=plt.subplots(1,2,figsize=(10.5,4.6))
for ax,values,title in zip(axes,[cm,cm/cm.sum(1,keepdims=True)],['Counts','Within-class proportion']):
 im=ax.imshow(values,cmap='Blues');ax.set_title(title)
 for i in range(7):
  for j in range(7):
   value=values[i,j];s=str(int(value)) if title=='Counts' else f'{value:.2f}'
   ax.text(j,i,s,ha='center',va='center',fontsize=8,color='white' if value>values.max()*.6 else 'black')
 ax.set_xticks(range(7),short,rotation=45,ha='right');ax.set_yticks(range(7),short)
 ax.set_xlabel('Predicted class');ax.set_ylabel('True class')
fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=300);plt.close(fig)
fig,ax=plt.subplots(figsize=(6.6,5.2));ax.imshow(cm,cmap='Blues')
for i in range(7):
 for j in range(7):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=10,color='white' if cm[i,j]>cm.max()*.6 else 'black')
ax.set_xticks(range(7),short,rotation=45,ha='right');ax.set_yticks(range(7),short)
ax.set_xlabel('Predicted class');ax.set_ylabel('True class')
fig.tight_layout();fig.savefig(out/'confusion_counts.png',dpi=300);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(7,3.7))
colors=['#496a81','#915340','#537650','#765990','#a37a2c','#457f83','#945e79']
for i,name in enumerate(short):
 fpr,tpr,_=roc_curve(y==i,p[:,i]);prec,rec,_=precision_recall_curve(y==i,p[:,i])
 axes[0].plot(fpr,tpr,color=colors[i],label=name);axes[1].plot(rec,prec,color=colors[i],label=name)
axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='One-versus-rest ROC')
axes[0].plot([0,1],[0,1],color='gray',linestyle='--',linewidth=.7)
axes[1].set(xlabel='Recall',ylabel='Precision',title='Precision-recall');axes[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(out/'roc_pr.png',dpi=300);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(7,3.4))
for tag,color in [('weighted_42','#496a81'),('unweighted_42','#915340')]:
 hist=json.loads((base/tag/'history.json').read_text());epoch=[v['epoch'] for v in hist]
 axes[0].plot(epoch,[v['training_loss_batch_mean'] for v in hist],label=tag,color=color)
 axes[1].plot(epoch,[v['validation_macro_f1'] for v in hist],label=tag,color=color)
axes[0].set(xlabel='Epoch',ylabel='Batch-mean training loss');axes[1].set(xlabel='Epoch',ylabel='Validation macro F1')
axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=300);plt.close(fig)
# Primary recipe and matched ablation use a fixed representative seed, not the best test score.
groups=np.array([r['group_id'] for r in rows]);ids=sorted(set(groups));members={g:np.flatnonzero(groups==g) for g in ids}
rng=np.random.default_rng(2026);differences=[]
pred,pred2=p.argmax(1),p2.argmax(1)
for _ in range(2000):
 sample=np.concatenate([members[g] for g in rng.choice(ids,len(ids),replace=True)])
 differences.append([accuracy_score(y[sample],pred[sample])-accuracy_score(y[sample],pred2[sample]),
 f1_score(y[sample],pred[sample],labels=range(7),average='macro',zero_division=0)-f1_score(y[sample],pred2[sample],labels=range(7),average='macro',zero_division=0)])
paired={'scope':'weighted minus unweighted, fixed seed 42, conditional on fitted heads','repetitions':2000,'bootstrap_seed':2026,
 'accuracy_difference':float(accuracy_score(y,pred)-accuracy_score(y,pred2)),
 'macro_f1_difference':float(f1_score(y,pred,labels=range(7),average='macro')-f1_score(y,pred2,labels=range(7),average='macro')),
 'interval_columns':['accuracy_difference','macro_f1_difference'],'group_bootstrap_95_ci':np.quantile(differences,[.025,.975],axis=0).tolist()}
(out/'paired_ablation.json').write_text(json.dumps(paired,indent=2))
manifest={r['path']:r for r in csv.DictReader(open('complete_prepared/manifest.csv'))};seen=set();canonical=[]
for i,r in enumerate(rows):
 digest=manifest[r['path']]['sha256_pixels']
 if digest not in seen:seen.add(digest);canonical.append(i)
sensitivity={'raw_test_images':len(rows),'unique_pixel_hashes_in_test':len(canonical),'content_groups_in_test':len(ids),
 'scope':'one retained prediction per exact decoded pixel hash; no retraining; near-copy links are not collapsed',
 'metrics':metrics(y[canonical],p[canonical])}
(out/'exact_dedup_sensitivity.json').write_text(json.dumps(sensitivity,indent=2))
train=[r for r in manifest.values() if r['split']=='train'];majority=int(np.bincount([int(r['label']) for r in train],minlength=7).argmax())
majority_prob=np.zeros((len(y),7));majority_prob[:,majority]=1
(out/'majority_baseline.json').write_text(json.dumps({'predicted_class':CLASSES[majority],
 'selection':'most frequent training label','test_metrics':metrics(y,majority_prob)},indent=2))
off=cm.copy();np.fill_diagonal(off,0);pairs=[]
for flat in np.argsort(off.ravel())[::-1][:5]:
 i,j=np.unravel_index(flat,off.shape);pairs.append({'true':CLASSES[i],'predicted':CLASSES[j],'count':int(off[i,j])})
(out/'error_pairs.json').write_text(json.dumps(pairs,indent=2))
print('RESULT ANALYSIS COMPLETE',json.dumps(paired), 'dedup images',len(canonical),flush=True)
