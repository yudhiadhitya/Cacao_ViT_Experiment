from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import json,hashlib,argparse
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from scipy.fft import dctn
p=argparse.ArgumentParser();p.add_argument('--root',default='complete_cacao');p.add_argument('--out',default='complete_audit')
args=p.parse_args();root=Path(args.root);paths=sorted(root.glob('*/*.png'))
def inspect(path):
 with Image.open(path) as opened:
  image=ImageOps.exif_transpose(opened).convert('RGB');pix=np.array(image)
  digest=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
  low=dctn(np.asarray(image.convert('L').resize((32,32)),dtype=float),norm='ortho')[:8,:8].ravel()[1:]
  phash=sum(int(v)<<i for i,v in enumerate(low>np.median(low)))
  return {'path':path.as_posix(),'class':path.parent.name,'width':image.width,'height':image.height,
    'pixel_sha256':digest,'phash':str(phash),'mean_intensity':float(pix.mean())}
with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(inspect,paths))
exact=defaultdict(list);similar=defaultdict(list)
for r in records:exact[r['pixel_sha256']].append(r);similar[r['phash']].append(r)
duplicates=[v for v in exact.values() if len(v)>1]
conflicts=[v for v in duplicates if len(set(r['class'] for r in v))>1]
candidates=[v for v in similar.values() if len(v)>1 and len(set(r['pixel_sha256'] for r in v))>1]
out=Path(args.out);out.mkdir(exist_ok=True)
(out/'image_records.json').write_text(json.dumps(records,indent=2))
summary={'total_images':len(records),'class_counts':dict(Counter(r['class'] for r in records)),
 'unique_pixel_hashes':len(exact),'exact_duplicate_groups':len(duplicates),'exact_duplicate_images':sum(len(v) for v in duplicates),
 'cross_label_exact_conflict_groups':len(conflicts),'same_phash_candidate_groups':len(candidates),
 'width_min':min(r['width'] for r in records),'width_max':max(r['width'] for r in records),
 'height_min':min(r['height'] for r in records),'height_max':max(r['height'] for r in records),
 'physical_identifiers_available':False,'sampling_scope':'image level; physical bean and batch independence unverified'}
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'duplicate_groups.json').write_text(json.dumps(duplicates,indent=2))
(out/'near_candidate_groups.json').write_text(json.dumps(candidates,indent=2));print(summary,flush=True)
for kind,groups in [('duplicates',duplicates),('near_candidates',candidates)]:
 for offset in range(0,len(groups),12):
  selected=groups[offset:offset+12];canvas=Image.new('RGB',(1000,160*len(selected)),'white');draw=ImageDraw.Draw(canvas)
  for row,group in enumerate(selected):
   for col,r in enumerate(group[:5]):
    with Image.open(r['path']) as im:canvas.paste(ImageOps.contain(im.convert('RGB'),(185,120)),(col*200,row*160))
    draw.text((col*200,row*160+125),r['class'],fill='black');draw.text((col*200,row*160+140),Path(r['path']).name,fill='black')
  canvas.save(out/f'{kind}_{offset//12}.png')
