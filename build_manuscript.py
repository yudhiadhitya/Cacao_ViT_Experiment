from pathlib import Path
import re, json, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

BASE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--source',default='manuscript.md');parser.add_argument('--output',default='Cacao_ViT_IMRAD_Protocol.docx')
args=parser.parse_args()
OUT=BASE/'output'; OUT.mkdir(exist_ok=True)
figdir=BASE/'figures';figdir.mkdir(exist_ok=True)
counts=[1187,1046,426,822,916,1776,1255]
labels=['Whole beans','Broken beans','Bean fractions','Skin-damaged beans','Fermented beans','Unfermented beans','Moldy beans']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,ax=plt.subplots(figsize=(7,3.2))
bars=ax.barh(labels[::-1],counts[::-1],color='#506f83',height=.62)
ax.bar_label(bars,padding=4,fmt='%d',fontsize=9)
ax.set_xlim(0,2050); ax.set_xlabel('Published image count')
ax.spines[['right','top','left']].set_visible(False)
ax.tick_params(axis='y',length=0); ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
fig.tight_layout();fig.savefig(figdir/'published_distribution.png',dpi=300);plt.close(fig)

doc=Document();section=doc.sections[0]
section.page_width=Inches(8.5);section.page_height=Inches(11)
section.top_margin=section.bottom_margin=Inches(.82)
section.left_margin=section.right_margin=Inches(.9)
section.footer_distance=Inches(.35)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
 style=doc.styles[name];style.font.name='Times New Roman';style.font.color.rgb=RGBColor(0,0,0)
 # Strip inherited theme fonts and paragraph rules from the runtime's default template.
 for element in style.element.iter():
  if element.tag==qn('w:rFonts'):
   for attr in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme','csTheme']:
    element.attrib.pop(qn('w:'+attr),None)
 for element in list(style.element.iter(qn('w:pBdr'))):
  element.getparent().remove(element)
 for element in list(style.element.iter(qn('w:contextualSpacing'))):
  element.getparent().remove(element)
normal=doc.styles['Normal'];normal.font.size=Pt(11.5)
normal.paragraph_format.line_spacing=1.15;normal.paragraph_format.space_after=Pt(6)
normal.paragraph_format.widow_control=True
doc.styles['Title'].font.size=Pt(18);doc.styles['Title'].font.bold=True
doc.styles['Title'].paragraph_format.space_after=Pt(8)
doc.styles['Heading 1'].font.size=Pt(13);doc.styles['Heading 1'].font.bold=True
doc.styles['Heading 1'].paragraph_format.space_before=Pt(14)
doc.styles['Heading 1'].paragraph_format.space_after=Pt(6)
doc.styles['Heading 2'].font.size=Pt(11.5);doc.styles['Heading 2'].font.bold=True
doc.styles['Heading 2'].paragraph_format.space_before=Pt(10)
doc.styles['Heading 2'].paragraph_format.space_after=Pt(5)
doc.styles['Caption'].font.size=Pt(10)
doc.styles['Caption'].paragraph_format.space_after=Pt(6)
for stylename in ['Heading 1','Heading 2']:
 doc.styles[stylename].paragraph_format.keep_with_next=True

footer=section.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
run=footer.add_run();run.font.size=Pt(9)
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');run._r.addnext(field)
doc.core_properties.title=(BASE/args.source).read_text().splitlines()[0].removeprefix('# ')
doc.core_properties.subject='One column IMRAD manuscript'
doc.core_properties.author=''

def table(lines):
 rows=[[part.strip() for part in line.strip().strip('|').split('|')] for line in lines]
 rows=[rows[0]]+rows[2:]
 n=len(rows[0]);t=doc.add_table(rows=0,cols=n);t.autofit=False
 widths={2:[2.25,4.45],4:[.55,3.05,1.45,1.65],5:[2.25,.85,.8,.8,2.0],6:[1.2,.55,.85,1.4,1.4,1.3]}[n]
 if n==4 and rows[0][0]=='Model':widths=[1.9,1.65,1.55,1.6]
 if n==4 and rows[0][0]=='Condition':widths=[2.,1.55,1.5,1.65]
 if n==5 and rows[0][1]=='Full archive':widths=[1.75,1.2,1.25,1.25,1.25]
 if n==5 and rows[0][1]=='Test images':widths=[2.45,.9,1.12,1.12,1.11]
 if n==5 and rows[0][0]=='Loss':widths=[1.,1.5,1.65,1.3,1.25]
 for j,width in enumerate(widths):t.columns[j].width=Inches(width)
 for i,row in enumerate(rows):
  cells=t.add_row().cells
  trpr=cells[0]._tc.getparent().get_or_add_trPr();nosplit=OxmlElement('w:cantSplit');trpr.append(nosplit)
  if i==0:
   repeat=OxmlElement('w:tblHeader');trpr.append(repeat)
  for j,text in enumerate(row):
   cell=cells[j];cell.width=Inches(widths[j]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   pr=cell._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
   for side in ['top','left','bottom','right']:
    edge=OxmlElement('w:'+side);edge.set(qn('w:val'),'single');edge.set(qn('w:sz'),'4');edge.set(qn('w:color'),'D9D9D9');borders.append(edge)
   pr.append(borders);margin=OxmlElement('w:tcMar')
   for side,value in [('top',70),('bottom',70),('left',90),('right',90)]:
    el=OxmlElement('w:'+side);el.set(qn('w:w'),str(value));el.set(qn('w:type'),'dxa');margin.append(el)
   pr.append(margin)
   if i==0:
    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E7E7E7');pr.append(shade)
   p=cell.paragraphs[0];p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.05
   numeric=text.replace(',','').replace('.','').isdigit()
   p.alignment=WD_ALIGN_PARAGRAPH.CENTER if numeric or (n==5 and j>0) else WD_ALIGN_PARAGRAPH.LEFT
   r=p.add_run(text);r.font.name='Times New Roman';r.font.size=Pt(10);r.bold=i==0
 doc.add_paragraph().paragraph_format.space_after=Pt(0)

lines=(BASE/args.source).read_text().splitlines();i=0;references=False
while i<len(lines):
 line=lines[i].strip()
 if not line:i+=1;continue
 if line.startswith('|'):
  data=[]
  while i<len(lines) and lines[i].strip().startswith('|'):data.append(lines[i]);i+=1
  table(data);continue
 if line.startswith('# '):
  p=doc.add_paragraph(line[2:],'Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
 elif line.startswith('## '):
  doc.add_paragraph(line[3:],'Heading 1');references=line=='## References'
 elif line.startswith('### '):doc.add_paragraph(line[4:],'Heading 2')
 elif line=='Experimental protocol and software verification':
  p=doc.add_paragraph(line,'Subtitle');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
  for r in p.runs:r.italic=True;r.font.size=Pt(11)
 elif line=='[FIGURE_DISTRIBUTION]':
  p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
  p.add_run().add_picture(str(figdir/'published_distribution.png'),width=Inches(6.3))
 elif line in ['[FIGURE_LEARNING]','[FIGURE_CONFUSION]','[FIGURE_ROCPR]']:
  files={'[FIGURE_LEARNING]':'learning_curves.png','[FIGURE_CONFUSION]':'confusion_counts.png','[FIGURE_ROCPR]':'roc_pr.png'}
  p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
  p.add_run().add_picture(str(BASE/'actual_frozen_msn_small/analysis'/files[line]),width=Inches(5.4 if line=='[FIGURE_CONFUSION]' else 6.3))
 elif re.match(r'^(Table|Figure) [0-9]+\.',line):
  p=doc.add_paragraph(line,'Caption');p.paragraph_format.keep_with_next=line.startswith('Table ')
 elif references:
  p=doc.add_paragraph(line);p.paragraph_format.left_indent=Inches(.28);p.paragraph_format.first_line_indent=Inches(-.28)
  p.paragraph_format.line_spacing=1.05;p.paragraph_format.space_after=Pt(4)
  for r in p.runs:r.font.size=Pt(10.5)
 else:
  p=doc.add_paragraph(line);p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
 i+=1
doc.save(OUT/args.output)
print('Created',OUT/args.output)
