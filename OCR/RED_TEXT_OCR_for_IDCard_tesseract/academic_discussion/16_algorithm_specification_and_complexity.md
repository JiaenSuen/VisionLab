# Algorithm Specification and Computational Complexity

## 1. Formal Pipeline

Given a document image `I` and normalized region of interest `R`, the pipeline returns a set of sequence crops `C = {C1, ..., Cn}` for downstream OCR. The computation is decomposed into localization and grouping.

### Algorithm A — Local Red-Contrast Localization

**Input:** BGR image `I`, ROI `R`, Gaussian scale `sigma`, component thresholds `(Amin, Wmin, Hmin)`.

1. Convert `R` to integer image coordinates and crop `IR = crop(I, R)`.
2. Convert `IR` to CIE Lab and extract the `a` channel `A`.
3. Compute local baseline `B = Gaussian(A, sigma)`.
4. Compute positive local red contrast `D = max(A - B, 0)`.
5. Estimate Otsu threshold `t` using only `D` inside the ROI.
6. Create raw mask `M = 1[D > t]`.
7. Create detection mask `Md = close(M, 2x2)`.
8. Label 8-connected components in `Md`.
9. Reject components with area `< Amin`, width `< Wmin`, or height `< Hmin`.
10. Translate accepted boxes from ROI coordinates back to global coordinates.

**Output:** raw mask `M`, detection mask `Md`, accepted component boxes `B = {b1, ..., bk}`.

### Algorithm B — Sequence Grouping

**Input:** component boxes `B`, maximum horizontal gap `g`, minimum vertical overlap `v`.

1. Create one graph node per box.
2. For each pair `(bi, bj)`, evaluate text-line compatibility using vertical overlap and center alignment.
3. If compatible and the horizontal gap is `<= g`, connect the pair.
4. Compute graph connected components using union-find.
5. Replace each graph component with the minimum enclosing rectangle of its member boxes.
6. Expand every rectangle by padding while clipping to image boundaries.
7. Crop from the original image `I`.
8. Resize each crop to target height `H` while preserving aspect ratio.

**Output:** `n` variable-width sequence images suitable for sequence OCR input.

## 2. Why This Decomposition Is Important

The first algorithm estimates *where red text evidence exists*. The second estimates *which pieces likely belong to the same visual sequence*. Neither stage attempts semantic decoding. This decomposition prevents a common design mistake in OCR systems: requiring a low-level binary segmentation to solve the high-level problem of character identity and sequence length.

## 3. Complexity

Let the ROI contain `P` pixels and let `k` connected-component boxes survive filtering. Color conversion, Gaussian filtering, thresholding, morphology, and connected-component labeling are approximately `O(P)` with implementation-dependent constants. The current grouping function compares every pair of boxes, producing `O(k^2)` compatibility checks. Union-find operations are effectively near-linear after those pairwise tests. For the current small ROI, `k` is small and the quadratic term is negligible.

If the method is extended to dense text pages, the grouping stage should avoid all-pairs comparison. Sorting boxes by x-coordinate and only checking neighbors within a spatial window can reduce practical complexity substantially. A spatial index or sweep-line method is another option.

## 4. Parameter Semantics

The current parameters are expressed in pixels and therefore encode assumptions about image scale. `sigma=3`, the 2x2 closing kernel, `Amin=8`, `Hmin=5`, `max_gap=8`, and padding 4 are meaningful only relative to the present character size. A more general algorithm should estimate a reference scale `s` from median component height or ROI height and express gap, padding, and area thresholds as functions of `s`.

## 5. Reproducibility Constraint

Algorithm A must preserve the order `ROI -> feature -> adaptive threshold`. Moving ROI restriction after Otsu is not an equivalent implementation because it changes the pixel population used to estimate `t`. This ordering constraint should be treated as part of the formal algorithm specification.

## Recognition-Crop Expansion

After a sequence group `B = (x, y, w, h)` is produced, the recognition crop is expanded without modifying the segmentation. The adaptive rule uses

```text
pad_x = max(P_min, round(r_x * h))
pad_y = max(P_min, round(r_y * h))
```

with current defaults `P_min = 4`, `r_x = 0.35`, and `r_y = 0.20`. Padding is clipped to the image boundary. The use of text height rather than image width makes the margin approximately scale-aware while remaining computationally constant-time per group.

## Adaptive Grouping Gap

For component set `C`, let `h_med` be the median component height. The effective horizontal gap is

```text
g = max(g_min, round(r_g * h_med))
```

with current defaults `g_min = 8` and `r_g = 1.0`. Pairwise grouping remains graph-based and therefore `O(n^2)` in the number of accepted component boxes. For the small number of components in the constrained ROI, this cost is negligible. A sweep-line or spatial index could reduce complexity for substantially larger text regions.
