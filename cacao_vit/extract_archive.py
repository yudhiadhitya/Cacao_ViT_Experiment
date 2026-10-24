"""Extract only image entries from the supplied cacao archive into canonical folders."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from collections import Counter
MAPPING={'bean_fraction':'bean_fractions','broken_beans':'broken_beans','fermented':'fermented_beans',
 'moldy':'moldy_beans','skin_damaged':'skin_damaged_beans','unfermented':'unfermented_beans','whole_beans':'whole_beans'}
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',required=True);p.add_argument('--out',default='complete_cacao');args=p.parse_args()
root=Path(args.out);root.mkdir(parents=True,exist_ok=False);seen=set();counts=Counter()
with zipfile.ZipFile(args.archive) as z:
 if z.testzip() is not None:raise ValueError('Archive CRC failure')
 for item in z.infolist():
  if not item.filename.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp')):continue
  relative=Path(item.filename)
  if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe archive member')
  classname=MAPPING[relative.parent.name];destination=root/classname/relative.name
  if destination in seen:raise ValueError('Repeated archive filename')
  seen.add(destination);destination.parent.mkdir(exist_ok=True);destination.write_bytes(z.read(item));counts[classname]+=1
print(json.dumps({'image_count':sum(counts.values()),'counts':dict(counts),
 'archive_sha256':hashlib.sha256(Path(args.archive).read_bytes()).hexdigest()},indent=2))
