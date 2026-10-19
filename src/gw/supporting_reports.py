"""Small-sample mechanism and distribution-fit summaries from original records."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from . import classification_report as classification
from .presentation import sha256


def checked(root, relative):
    path = Path(root)/relative
    manifest = json.loads((Path(root)/'artifacts/manifest.json').read_text())['source_files']
    if sha256(path) != manifest[relative]['sha256']:
        raise ValueError(f'Source checksum mismatch: {relative}')
    return path


def toy(root):
    confidence = pd.read_csv(checked(root,'results/toy1/confidence.csv')).set_index('seed')
    response = pd.read_csv(checked(root,'results/toy1/response.csv')).set_index('seed')
    if confidence.index.tolist()!=list(range(100,130)) or not confidence.index.equals(response.index):
        raise ValueError('Thirty paired seeds required')
    raw = pd.DataFrame({'Bias · initial':confidence.untrained_confidence_bias,
                        'Bias · warm-up':confidence.warmup_confidence_bias,
                        'Response · initial':response.initial_mean_log_response,
                        'Response · warm-up':response.warmup_mean_log_response})
    table = pd.DataFrame(index=['Confidence bias ↓','Mean log response ↓'])
    for i,label in enumerate(table.index):
        before,after=raw.iloc[:,2*i],raw.iloc[:,2*i+1]
        digits=4 if i==1 else 5
        table.loc[label,'Initialization']=f'{before.mean():.{digits}f} ± {before.std(ddof=1):.{digits}f}'
        table.loc[label,'Warm-up']=f'{after.mean():.{digits}f} ± {after.std(ddof=1):.{digits}f}'
        table.loc[label,'Reduced seeds']=f'{int((after<before).sum())}/30'
    return raw,table


def toy_inference(root):
    from .toy_model import ToyModel, confidence_metrics, projected_response_metrics, unit_directions
    raw,_=toy(root)
    points=np.random.default_rng(20260709).uniform(-10,10,size=(10000,2))
    directions=unit_directions(32)
    rows=[]
    for seed in range(100,130):
        for phase in ['initial','warmup']:
            path=checked(root,f'artifacts/toy1/{seed}/{phase}.npz')
            with np.load(path,allow_pickle=False) as z:
                model=ToyModel([z[f'weight_{i}'].copy() for i in range(3)],[z[f'bias_{i}'].copy() for i in range(3)])
            bias=confidence_metrics(model,points)['confidence_bias']
            response=projected_response_metrics(model,points,directions,epsilon=1e-12)['mean_log_response']
            col='initial' if phase=='initial' else 'warm-up'
            delta=max(abs(bias-raw.loc[seed,f'Bias · {col}']),abs(response-raw.loc[seed,f'Response · {col}']))
            if delta>1e-9:
                raise ValueError(f'Toy checkpoint measurement mismatch: seed {seed}, {phase}, {delta}')
            rows.append({'Seed':seed,'Phase':phase,'Confidence bias':bias,'Mean log response':response,'Max difference':delta})
    return pd.DataFrame(rows)


def toy_plot(raw,output):
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.4),layout='constrained')
    for i,(axis,title) in enumerate(zip(axes,['Confidence bias ↓','Mean log response ↓'])):
        before,after=raw.iloc[:,2*i],raw.iloc[:,2*i+1]
        for a,b in zip(before,after):
            axis.plot([0,1],[a,b],color='#9eafb9',alpha=.45,linewidth=.8)
        axis.scatter(np.zeros(len(before)),before,color='#73808a',s=16)
        axis.scatter(np.ones(len(after)),after,color='#226b91',s=16)
        axis.set_xticks([0,1],['Initialization','Warm-up'])
        axis.set_title(title,loc='left',fontsize=13)
        axis.spines[['top','right']].set_visible(False)
    fig.savefig(output,dpi=180,bbox_inches='tight')
    return fig


def interaction(root, directory=None):
    config=classification.read(root,'noise')
    runs,_,_=classification.calculate(root,config,directory)
    pivot=runs.pivot(index='Seed',columns=['Dataset','Method'],values='ECE (%)')
    if len(pivot)!=10 or pivot.isna().any().any():
        raise ValueError('Ten complete paired noise seeds required')
    raw=pd.DataFrame({'Gaussian domain':pivot[('Gaussian','GW')]-pivot[('Gaussian','Gaussian')],
                      'LogGamma domain':pivot[('LogGamma','GW')]-pivot[('LogGamma','Gaussian')]})
    raw['Interaction']=raw['LogGamma domain']-raw['Gaussian domain']
    rows=[]
    for column in raw:
        values=raw[column]
        margin=t.ppf(.975,9)*values.std(ddof=1)/np.sqrt(10)
        rows.append({'Comparison':column,'Mean (pp)':values.mean(),'CI low':values.mean()-margin,
                     'CI high':values.mean()+margin,'Negative seeds':int((values<0).sum())})
    return raw,pd.DataFrame(rows).set_index('Comparison')


def interaction_plot(raw,summary,output):
    fig,axis=plt.subplots(figsize=(7.8,3.3),layout='constrained')
    for i,col in enumerate(raw):
        axis.scatter(raw[col],i+np.linspace(-.1,.1,len(raw)),s=18,alpha=.4,color='#73808a')
        row=summary.loc[col]
        axis.errorbar(row['Mean (pp)'],i,xerr=[[row['Mean (pp)']-row['CI low']],[row['CI high']-row['Mean (pp)']]],
                      fmt='D',color='#226b91',capsize=4)
    axis.set_yticks(range(len(raw.columns)),raw.columns)
    axis.invert_yaxis()
    axis.axvline(0,color='#9eafb9',linestyle='--',linewidth=1)
    axis.set_xlabel('Paired ECE difference (pp)')
    axis.spines[['top','right','left']].set_visible(False)
    fig.savefig(output,dpi=180,bbox_inches='tight')
    return fig


def fit(root):
    sar=json.loads(checked(root,'results/fit/sar_fits.json').read_text())
    oct_rows=json.loads(checked(root,'results/fit/oct_fits.json').read_text())
    representative=json.loads(checked(root,'results/fit/representative.json').read_text())['rows']
    rows=[]
    sources=[('OpenSARShip2','opensarship2_grd3','percentile',sar),
             ('Schwegmann SAR','sar_dataset_window21','percentile',sar),
             ('FUSAR-Ship','fusar_ship_paper5_proxy','stored',sar),
             ('OCTDL','oct_background','model_gray',oct_rows)]
    for label,dataset,coordinate,source in sources:
        for family in ['GeneralizedGamma','Gaussian','Uniform','Laplace']:
            bank=source if family=='GeneralizedGamma' else representative
            fit_coordinate=coordinate if family=='GeneralizedGamma' else 'model_gray'
            selected=[r for r in bank if r['dataset']==dataset and r['coordinate']==fit_coordinate and r['family']==family]
            if len(selected)!=5 or sorted(r['seed'] for r in selected)!=list(range(91501,91506)):
                raise ValueError(f'Five seeded group-split records required: {label}, {family}')
            rows.append({'Dataset':label,'Family':{'GeneralizedGamma':'Generalized Gamma'}.get(family,family),
                         'NLL ↓':np.mean([np.inf if r['nll'] is None else r['nll'] for r in selected]),
                         'KS ↓':np.mean([r['ks'] for r in selected])})
    return pd.DataFrame(rows).set_index(['Dataset','Family'])


def fit_plot(table,output):
    selected=table['KS ↓'].unstack('Family')
    fig,axis=plt.subplots(figsize=(9,3.8),layout='constrained')
    for i,family in enumerate(selected):
        axis.scatter(np.arange(len(selected))+(i-1.5)*.13,selected[family],s=45,
                     label=family,color={'Generalized Gamma':'#226b91','Gaussian':'#73808a',
                                         'Laplace':'#b18a61','Uniform':'#9faeb9'}[family])
    axis.set_xticks(range(len(selected)),selected.index)
    axis.set_ylabel('KS ↓')
    axis.legend(frameon=False,ncol=2)
    axis.spines[['top','right']].set_visible(False)
    fig.savefig(output,dpi=180,bbox_inches='tight')
    return fig
