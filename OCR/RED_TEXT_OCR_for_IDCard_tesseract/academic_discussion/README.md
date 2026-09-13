# Academic Discussion Archive

This directory records the reasoning path behind the red-text OCR project: initial hypotheses, image-processing choices, failed variants, regression causes, successful parameter interactions, sequence-grouping logic, recognition-crop design, Tesseract integration, and reproducible experimental protocol. The objective is to preserve not only the current implementation but also a reusable method for solving future low-resolution OCR problems without blind parameter search.

The active project now separates four stages: **signal localization**, **geometric proposal generation**, **recognition crop construction**, and **OCR**. The successful detector is frozen as a baseline; recognition currently uses Tesseract so that OCR quality can be measured without introducing a new training problem. Historical CRNN material is retained under `archive/previous_crnn_iteration/` because it documents an important negative transfer experiment, but CRNN code and weights are no longer part of the executable repository.

## Reading Order

1. [`01_research_problem_and_hypotheses.md`](01_research_problem_and_hypotheses.md) — problem definition and working hypotheses.
2. [`02_experiment_evolution_and_version_history.md`](02_experiment_evolution_and_version_history.md) — experiment/version history.
3. [`03_red_segmentation_analysis.md`](03_red_segmentation_analysis.md) — HSV, Lab local contrast, ROI ordering, and threshold behavior.
4. [`04_connected_components_analysis.md`](04_connected_components_analysis.md) — why connected components are useful and where they fail.
5. [`05_failure_analysis_and_root_causes.md`](05_failure_analysis_and_root_causes.md) — failure taxonomy and causal debugging.
6. [`06_success_case_key_factors.md`](06_success_case_key_factors.md) — why the successful prototype worked.
7. [`07_sequence_grouping_and_ocr_transition.md`](07_sequence_grouping_and_ocr_transition.md) — move from components to sequence-level OCR proposals.
8. [`08_image_processing_knowledge_base.md`](08_image_processing_knowledge_base.md) — reusable image-processing principles.
9. [`09_experimental_protocol_and_metrics.md`](09_experimental_protocol_and_metrics.md) — quantitative protocol and metrics.
10. [`10_ablation_plan_and_future_experiments.md`](10_ablation_plan_and_future_experiments.md) — controlled future experiments.
11. [`11_reproducibility_and_debugging_protocol.md`](11_reproducibility_and_debugging_protocol.md) — stage-by-stage debugging protocol.
12. [`12_mini_paper.md`](12_mini_paper.md) — compact paper-style synthesis.
13. [`13_literature_notes.md`](13_literature_notes.md) — literature notes and references.
14. [`14_decision_log.md`](14_decision_log.md) — engineering/research decisions.
15. [`15_future_project_playbook.md`](15_future_project_playbook.md) — from-zero workflow for future projects.
16. [`16_algorithm_specification_and_complexity.md`](16_algorithm_specification_and_complexity.md) — algorithm specification and complexity.
17. [`17_bbox_padding_ablation.md`](17_bbox_padding_ablation.md) — tight detection versus expanded recognition crops.
18. [`18_scale_aware_grouping_ablation.md`](18_scale_aware_grouping_ablation.md) — fixed-pixel versus scale-normalized grouping.
19. [`21_tesseract_ocr_integration.md`](21_tesseract_ocr_integration.md) — current OCR integration and preprocessing ablation.
20. [`module_analysis/`](module_analysis/) — module-level implementation analysis.
21. [`../experiments/results/SUMMARY.md`](../experiments/results/SUMMARY.md) — generated experimental evidence.
22. [`archive/previous_crnn_iteration/`](archive/previous_crnn_iteration/) — historical CRNN integration and domain-shift record.

## Current Evidence Status

The repository contains a controlled two-image case study. Within those examples, the modular detector reproduces the successful historical implementation; adaptive/four-pixel recognition padding recovers the full current weak-stroke support proxy; threshold relaxation destabilizes component topology; a fixed eight-pixel grouping distance is scale-sensitive; and grayscale 4x upsampling enables Tesseract to achieve 2/2 exact sequence matches with mean CER 0.0000. Raw color and hard Otsu inputs perform worse under the same crop geometry.

These observations are useful causal engineering findings, not population-level OCR claims. A larger labeled dataset is still required to measure robustness across resolution, blur, compression, illumination, font variation, and document capture conditions. Future claims should distinguish localization metrics from recognition metrics so that an OCR error is not incorrectly attributed to segmentation, or vice versa.
