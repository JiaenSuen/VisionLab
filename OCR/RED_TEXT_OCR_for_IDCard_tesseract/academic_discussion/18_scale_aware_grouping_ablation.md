# Scale-Aware Sequence Grouping

## Problem

The original sequence grouping used a fixed horizontal distance of eight pixels. This value worked on the smaller example, but the larger image revealed a hidden scale assumption: text representing the same semantic line occupied more pixels, so gaps that were normal within the line exceeded the fixed threshold. The result was not a segmentation failure. The same connected components were available, but the grouping stage split one intended text sequence into several independent crops.

## Ablation

Three settings were compared using the unchanged component detector: fixed 8 pixels, fixed 10 pixels, and an adaptive rule defined as `max(8, median_component_height * 1.0)`. On the smaller example, all three settings produced three groups and preserved the largest line width of 129 pixels. On the larger example, fixed 8 pixels produced five groups and the largest group width was only 76 pixels. Fixed 10 pixels produced three groups with a largest width of 142 pixels. The adaptive rule estimated 10 pixels from the observed component scale and reproduced the three-group result.

## Interpretation

The finding is modest but generalizable: pixel distances should be normalized by an observable scale whenever input resolution is not fixed. Character height is an inexpensive proxy because it is already available from connected-component statistics. The adaptive rule is still heuristic and does not establish semantic word boundaries. Its purpose is to make the proximity graph less sensitive to image resize scale while preserving the transitive grouping behavior: if A is close to B and B is close to C on the same line, all three can belong to one sequence even when A and C are not directly close.

## Remaining Risk

Increasing the gap can over-merge independent fields if they share a line. Therefore, the adaptive distance must remain coupled with vertical compatibility, ROI constraints, and future field-level priors. For a broader dataset, the next ablation should measure group completeness and group purity rather than only the number of groups.
