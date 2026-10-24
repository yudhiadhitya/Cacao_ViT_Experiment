from pathlib import Path
import csv,json,sys
import numpy as np
import torch
from PIL import Image,ImageOps
sys.path.insert(0,'cacao_vit')
from experiment import model,transform,seed,CLASSES
seed(42);base=Path('actual_frozen_msn_small')
records=list(csv.DictReader(open('complete_prepared/manifest.csv')));test=[r for r in records if r['split']=='test']
pred=list(csv.DictReader(open(base/'weighted_42/test_predictions.csv')))
head=torch.nn.Linear(384,7);head.load_state_dict(torch.load(base/'weighted_42/classifier.pt',weights_only=True))
net=model('msn_small',pretrained=True);net.classifier.load_state_dict(head.state_dict());net.eval()
errors=[]
for r in pred[:7]:
 with Image.open(Path('complete_cacao')/r['path']) as im:x=transform('msn_small',False)(ImageOps.exif_transpose(im).convert('RGB')).unsqueeze(0)
 with torch.no_grad():p=net(x).logits.softmax(-1).numpy()[0]
 expected=np.array([float(r[f'p_{i}']) for i in range(7)]);error=float(np.max(np.abs(p-expected)));errors.append(error)
 assert np.allclose(p,expected,atol=1e-5,rtol=1e-5)
 assert p.argmax()==int(r['prediction'])
report={'status':'passed','images_checked':len(errors),'max_absolute_probability_difference':max(errors),
 'method':'direct reconstructed ViT plus loaded trained head versus cached-feature prediction',
 'checkpoint':'weighted_42','cacao_training_executed':True,'classifier_parameters':sum(p.numel() for p in head.parameters()),
 'backbone_parameters':sum(p.numel() for p in net.base_model.parameters()),'model_parameters':sum(p.numel() for p in net.parameters())}
(base/'analysis/reconstruction_verification.json').write_text(json.dumps(report,indent=2));print(report)
