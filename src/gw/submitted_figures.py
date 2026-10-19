"""Four-figure submission gallery with explicit quantitative-component mapping."""
from pathlib import Path
import base64
import datetime
import html
import json
import shutil
from . import paper_figures as components
from .supporting_reports import checked

def catalog(root):
    records=json.loads(checked(root,'configs/paper_figures.json').read_text())['figures']
    if [r['number'] for r in records]!=[1,2,3,4]:raise ValueError('Expected four submitted figures')
    for row in records:
        for asset in row['assets']:
            path=checked(root,asset['path'])
            if path.stat().st_size!=asset['bytes']:raise ValueError('Figure size mismatch')
    return records

def paper_html(root,number):
    row=next(r for r in catalog(root) if r['number']==number)
    images=''.join('<img style="max-width:100%;height:auto" alt="Figure '+str(number)+'" src="data:image/png;base64,'+base64.b64encode(checked(root,a['path']).read_bytes()).decode()+'">' for a in row['assets'])
    return images+'<details><summary>Submitted caption (LaTeX source)</summary><pre style="white-space:pre-wrap">'+html.escape(row['caption_latex'])+'</pre></details>'

def run(root,number,mode='paper'):
    root=Path(root);row=next(r for r in catalog(root) if r['number']==number)
    if mode=='quantitative':
        if not row['quantitative_components']:raise ValueError('Schematic figure; use paper mode')
        return [components.run(root,n,mode='quantitative') for n in row['quantitative_components']]
    if mode!='paper':raise ValueError('Unknown mode')
    dest=root/'output/submitted_figures'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')/f'figure{number}'
    dest.mkdir(parents=True,exist_ok=False)
    for asset in row['assets']:shutil.copyfile(checked(root,asset['path']),dest/Path(asset['path']).name)
    (dest/'caption.tex').write_text(row['caption_latex']+'\n')
    (dest/'source.json').write_text(json.dumps(row,indent=2)+'\n')
    return dest
