# Submitted figure mapping

The default gallery follows the 18 October 2026 manuscript: four figures.
`configs/paper_figures.json` includes exact LaTeX captions, PDF page/crop bounds,
source PDF SHA256 and four image hashes. PDF crops preserve submitted labels.

| Submission figure | Earlier quantitative component IDs | Source panels |
|---|---|---|
| 1 | none | schematic input variations and probability contours |
| 2 | 2, 3 | initial states, directional responses, CIFAR trajectory/reliability |
| 3 | 4, 6 | optical-noise ECE, local fits and warm-up-family ECE |
| 4 | none | generalized warm-up framework schematic |

The historic component catalog and generator APIs remain in
`configs/quantitative_components.json` and `gw.paper_figures`; their identifiers
are not submitted figure numbers. `gw.submitted_figures` is the current gallery
and export interface. Quantitative regeneration produces separate components;
it does not claim exact recreation of the submitted composite layout.

The author confirmed that the paper illustrations were generated using GPT.
This provenance statement applies to illustrations; measured image pixels,
dataset samples and pretraining sources retain their existing provenance.
Additional sample metadata is retained at the author's request.
