"""Rerun all registered post-hoc base models in explicit FP32 on CUDA."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from gw import posthoc, posthoc_inference
from gw.classification_report import comparison, paper


if __name__ == '__main__':
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This verification requires CUDA')
    directory = posthoc_inference.run(ROOT, device='cuda')
    runs, mean, sd = posthoc.summarize(ROOT, directory)
    check = comparison(mean, sd, paper(ROOT, 's6_posthoc'))
    check.to_csv(directory/'paper_comparison.csv')
    refit = posthoc_inference.refit(ROOT, directory)
    _, refit_mean, refit_sd = posthoc.summarize(ROOT, refit)
    import numpy as np
    np.testing.assert_array_equal(mean.values, refit_mean.values)
    np.testing.assert_array_equal(sd.values, refit_sd.values)
    receipt = dict(status='complete', inference_device='cuda', inference_precision='fp32',
        checkpoint_inference_executed=True, base_models=6, calibration_outputs=len(runs),
        results=directory.relative_to(ROOT).as_posix(), refit=refit.relative_to(ROOT).as_posix(),
        paper_precision_matches=int(check.Matches.sum()), metrics_compared=len(check),
        same_input_refit_exact=True,
        ece_means={method:float(mean.loc[('OCTDL','ResNet-18',method), 'ECE (%)'])
                   for method in mean.index.get_level_values('Method')})
    (ROOT/'posthoc_fp32_verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)
