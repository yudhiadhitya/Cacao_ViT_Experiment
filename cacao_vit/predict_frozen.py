"""Reconstruct the trained frozen MSN-Small classifier for one image."""
import argparse,json
from pathlib import Path
import torch
from PIL import Image,ImageOps
from experiment import model,transform,CLASSES,seed

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--image',required=True)
 p.add_argument('--classifier',required=True);p.add_argument('--out')
 args=p.parse_args();seed(42);net=model('msn_small',pretrained=True)
 net.classifier.load_state_dict(torch.load(args.classifier,map_location='cpu',weights_only=True));net.eval()
 with Image.open(args.image) as im:input_tensor=transform('msn_small',False)(ImageOps.exif_transpose(im).convert('RGB')).unsqueeze(0)
 with torch.no_grad():prob=net(input_tensor).logits.softmax(-1)[0].numpy()
 result={'label':CLASSES[int(prob.argmax())],'probabilities':dict(zip(CLASSES,map(float,prob))),
  'scope':'frozen MSN-Small encoder and trained seven-class linear head'}
 if args.out:Path(args.out).write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()
