# Experiment Log Template

## Metadata

- Experiment ID:
- Date:
- Code version / commit:
- Dataset / image list:
- Research question:
- Hypothesis:

## Frozen Baseline

- ROI:
- Feature representation:
- Blur/local baseline:
- Threshold method:
- Morphology:
- Connected-component parameters:
- Grouping parameters:
- Recognition backend (if any):

## Single Experimental Change

Describe exactly one causal change relative to the baseline.

## Measurements

- Otsu/automatic threshold value(s):
- Raw component count:
- Accepted component count:
- Group count:
- Pixel precision/recall/IoU (if labeled):
- Character coverage:
- Fragmentation rate:
- Merge rate:
- Group completeness/purity:
- CER / exact-string accuracy:

## Observations

Record what changed visually and numerically without explaining why yet.

## Interpretation

State the proposed mechanism and distinguish it from direct observation.

## Failure Cases

List representative images and the first pipeline stage where the result diverged from expectation.

## Conclusion

State whether the hypothesis was supported, rejected, or remains inconclusive. Define the next controlled experiment.
