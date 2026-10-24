from pathlib import Path
from collections import defaultdict
import numpy as np,json,csv
from PIL import Image,ImageOps,ImageDraw
records=json.loads(Path('complete_audit/image_records.json').read_text())
byhash={}
for record in records:byhash.setdefault(record['pixel_sha256'],record)
unique=list(byhash.values());hashes=np.array([int(r['phash']) for r in unique],dtype=np.uint64)
parent=list(range(len(unique)))
def find(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
def norm(path):
 with Image.open(path) as opened:
  im=ImageOps.exif_transpose(opened).convert('RGB');a=np.asarray(im);yy,xx=np.where(a.max(2)>8)
  if len(xx):im=im.crop((int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)))
  return np.asarray(im.resize((64,64),Image.Resampling.BICUBIC),dtype=np.float64)/255
candidates=[];accepted=[];cached={}
for i,h in enumerate(hashes):
 distances=np.bitwise_count(hashes[i+1:]^h)
 for relative in np.flatnonzero(distances<=4):
  j=i+1+int(relative)
  if i not in cached:cached[i]=norm(unique[i]['path'])
  if j not in cached:cached[j]=norm(unique[j]['path'])
  a,b=cached[i],cached[j];rmse=float(np.sqrt(np.mean((a-b)**2)));corr=float(np.corrcoef(a.ravel(),b.ravel())[0,1])
  entry={'a':unique[i]['path'],'b':unique[j]['path'],'hamming':int(distances[relative]),'normalized_rmse':rmse,'normalized_correlation':corr}
  candidates.append(entry)
  if corr>=.98 and rmse<=.02:parent[find(j)]=find(i);accepted.append(entry)
index={r['pixel_sha256']:i for i,r in enumerate(unique)}
with open('complete_audit/content_groups.csv','w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['path','group_id']);writer.writeheader()
 for r in records:
  path=Path(r['path']).relative_to('complete_cacao').as_posix();writer.writerow({'path':path,'group_id':'content_'+str(find(index[r['pixel_sha256']]))})
Path('complete_audit/near_copy_audit.json').write_text(json.dumps({'scope':'content similarity grouping; physical identities not available',
 'hash_bits':63,'candidate_max_hamming':4,'accepted_min_correlation':.98,'accepted_max_rmse':.02,
 'candidate_pairs':len(candidates),'accepted_pairs':len(accepted),'content_groups':len(set(find(i) for i in range(len(unique)))),
 'accepted':accepted,'candidates':candidates},indent=2))
print('Near-copy candidates',len(candidates),'accepted',len(accepted),'content groups',len(set(find(i) for i in range(len(unique)))),flush=True)
for offset in range(0,len(accepted),12):
 selected=accepted[offset:offset+12];canvas=Image.new('RGB',(750,150*len(selected)),'white');draw=ImageDraw.Draw(canvas)
 for row,entry in enumerate(selected):
  for col,key in enumerate(['a','b']):
   path=Path(entry[key])
   with Image.open(path) as im:canvas.paste(ImageOps.contain(im.convert('RGB'),(220,110)),(col*350,row*150))
   draw.text((col*350,row*150+115),path.parent.name+'/'+path.name,fill='black')
 canvas.save(f'complete_audit/accepted_near_{offset//12}.png')
