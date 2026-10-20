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

The quantitative component catalog and generator APIs are in
`configs/quantitative_components.json` and `gw.paper_figures`.
`gw.submitted_figures` provides the four-figure gallery and export interface.
The component generators export the quantitative panels listed above.

The schematic illustrations were generated using GPT. Dataset images and
pretraining sources are documented in `data/README.md` and `docs/THIRD_PARTY.md`.
