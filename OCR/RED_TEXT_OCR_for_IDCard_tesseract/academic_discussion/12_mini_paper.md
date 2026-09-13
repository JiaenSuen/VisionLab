# A Lightweight Chromatic-Contrast, Component-Grouping, and Tesseract Pipeline for Red Alphanumeric OCR in Low-Resolution Documents

## Abstract

This study investigates an interpretable OCR pipeline for red alphanumeric text in low-resolution, semi-structured document images. Early HSV thresholding experiments damaged thin character strokes while retaining unrelated red structures. A more stable localization design was obtained by cropping the target region before statistical processing, converting it to CIE Lab, computing positive local contrast on the `a` channel, applying Otsu thresholding, repairing small gaps with a 2x2 closing operation, and extracting 8-connected components with conservative geometric filtering. A later modularization regression revealed that operation order is a critical part of the algorithm: moving the ROI after Otsu changed the threshold statistics, removing closing increased fragmentation, and loosening component filters admitted small artifacts. Because connected components do not correspond reliably to semantic characters, neighboring proposals are grouped into sequence-level regions. Detection boxes remain tight, while recognition crops are expanded from the original image to preserve anti-aliased edge information. Tesseract is used as the current OCR baseline. On two controlled examples, grayscale 4x upsampling produced 2/2 exact sequence matches with mean character error rate (CER) 0.0000, compared with 0/2 for raw color and 1/2 for Otsu-binarized recognition inputs. The study remains a small engineering case study, but it provides concrete evidence for separating localization geometry from recognition representation.

## 1. Introduction

OCR pipelines are often treated as a single recognition problem, but narrow document domains provide priors that can reduce complexity and improve interpretability. In the present task, the target is a red alphanumeric sequence located in an approximately known document region. This suggests a hybrid design: classical image processing performs localization and proposal formation, while an OCR engine handles sequence recognition.

The dominant difficulty is low spatial resolution. Anti-aliasing, resampling, and compression produce character-edge pixels that are mixtures of foreground and background. A strict color threshold can therefore remove valid stroke support. Morphological operators can either repair or worsen this problem depending on topology: opening can erase narrow strokes, while a small closing can reconnect threshold-induced gaps. Color alone is also insufficient because flags, logos, and security graphics may be genuinely red. The system therefore combines spatial restriction, local chromatic contrast, conservative topology repair, and geometric reasoning.

## 2. Localization Method

### 2.1 ROI-First Statistical Scope

Let the document image be `I` and the normalized target region be `R=(x1,y1,x2,y2)`. The region is cropped before feature extraction and adaptive threshold estimation. This order is essential because Otsu thresholding estimates a decision threshold from the histogram of its input. Applying the ROI only after Otsu does not merely move a mask; it changes the statistical population used to define the threshold and therefore changes the detector.

### 2.2 Local Red Contrast in CIE Lab

The ROI is converted from BGR to CIE Lab. The `a` channel encodes the green-red chromatic axis. A Gaussian-smoothed baseline is subtracted from the original channel:

\[
D(x,y)=\max(A(x,y)-G_\sigma(A)(x,y),0).
\]

This representation emphasizes pixels that are locally redder than their surroundings rather than globally satisfying a fixed red interval. Otsu thresholding converts the response into a binary localization mask.

### 2.3 Topological Repair and Components

A 2x2 morphological closing is applied only to the detection mask. Eight-connected-component labeling then produces proposal regions. The current baseline rejects components with area below 8 pixels, width below 2 pixels, or height below 5 pixels. These values are engineering parameters for the current image scale rather than universal constants.

## 3. Sequence Grouping and Recognition Crops

Connected components are not interpreted as guaranteed characters. One character may fragment into several regions, while multiple touching characters may become one region. The accepted components are therefore treated as nodes in a proximity graph. Horizontal distance and vertical compatibility define edges, and union-find creates transitive groups. Each group is converted into one enclosing rectangle.

A second rectangle is then constructed for recognition. The localization rectangle stays tight, but the OCR crop is expanded using fixed or character-height-adaptive padding and sampled from the original image. This two-representation design prevents uncertain edge pixels from changing component topology while still exposing them to the recognizer.

A controlled support-coverage experiment illustrates the difference. Across the two current examples, tight boxes retained 0.8262 mean estimated weak-stroke support with a minimum of 0.5797. Two-pixel padding increased mean coverage to 0.9853, and fixed four-pixel or adaptive padding reached 1.0000 on the current proxy.

## 4. Tesseract OCR Baseline

Tesseract is used as the active sequence recognizer. Each target group is treated as a single text line (`PSM 7`) with an English alphanumeric whitelist. Short red groups are skipped by a configurable aspect-ratio prior because the current ROI also contains small non-target red labels.

Three recognition representations were evaluated while keeping localization, grouping, and crop geometry constant: raw color, grayscale with 4x cubic upsampling, and grayscale with 4x upsampling followed by Otsu binarization. The two controlled examples have known target sequences: `A234567890` and `BTS4ever J-hope`.

| Recognition input | Exact matches | Mean CER |
|---|---:|---:|
| Raw color | 0/2 | 0.5666 |
| Grayscale + 4x upsampling | 2/2 | 0.0000 |
| Otsu binary + 4x upsampling | 1/2 | 0.0333 |

The grayscale result supports the hypothesis that low-resolution edge evidence should be preserved for recognition. Raw color leaves unnecessary background variation, while a second hard threshold can reshape weak strokes. This is consistent with the broader system design: binary decisions are useful for localization geometry, but recognition can benefit from graded intensity information.

## 5. Failure Analysis

The most important historical failure occurred during modularization. A version that appeared algorithmically similar produced many more false boxes because three details changed simultaneously: ROI application moved after full-image Otsu estimation, the 2x2 closing was removed, and geometric filters were loosened. Restoring the original order and parameters restored the expected behavior. This failure demonstrates that refactoring image-processing pipelines must be treated as an experimentally sensitive operation rather than a purely structural code change.

A second failure mode was the attempt to improve visually tight boxes by lowering the segmentation threshold. On the larger example, relaxing the threshold increased foreground support from 953 to 1606 pixels but also changed the sequence topology from three groups to six. More detected foreground therefore did not imply a better OCR interface. The successful correction was to freeze segmentation and expand only the recognition crop.

A third failure mode occurs at the OCR stage. Under identical crop geometry, raw and binary recognition inputs produced more errors than grayscale upsampling. This shows that end-to-end OCR errors can arise after correct localization and should not automatically be attributed to the detector.

## 6. Discussion

The current results support several practical principles. First, the scope of adaptive statistics is part of an algorithm definition. Second, morphology should be selected according to the observed topological error rather than used as generic denoising. Third, connected components are interpretable proposal regions but unreliable semantic character boundaries. Fourth, localization and recognition have different information requirements and should be optimized separately. Finally, OCR preprocessing should preserve uncertain edge evidence unless an ablation demonstrates that further binarization is beneficial.

The use of Tesseract in this iteration is deliberate. A previous historical CRNN checkpoint could be executed after modularization but did not transfer from its CAPTCHA training domain to the document domain. Removing it from the active runtime reduces confounding and establishes a practical non-trained OCR baseline. The historical negative result remains documented in the archive because it demonstrates the difference between successful software integration and successful model transfer.

## 7. Limitations and Future Work

The study currently contains only two controlled document examples, so the exact OCR result must not be generalized as benchmark-level evidence. A larger labeled dataset should vary scale, blur, JPEG quality, illumination, capture angle, font, and background security patterns. Ground-truth character strings, text-line boxes, and optionally pixel masks would enable CER, exact-match accuracy, localization coverage, group purity/completeness, fragmentation, and merge-rate analysis.

The next model-related experiment should compare the Tesseract baseline against a target-domain learned sequence recognizer under the **same crop protocol**. This prevents model comparisons from being confounded by different localization or padding strategies. Any future neural model should be evaluated only after a train/validation/test split is defined and the target vocabulary is explicit.

## 8. Conclusion

A lightweight classical front end can localize red alphanumeric text effectively when its assumptions match the document domain and its operations are executed in the correct order. The strongest current findings are that ROI-before-Otsu is statistically essential, minimal closing improves low-resolution topology, connected components should be treated as proposals, and recognition crops should be larger than localization boxes. The Tesseract experiment adds a new end-to-end result: grayscale 4x upsampling preserved sufficient low-resolution character evidence to achieve exact recognition on both controlled examples, while raw color and hard binarization were less reliable. The principal methodological contribution is therefore a separation of **where text is**, **how proposals are grouped**, and **what representation the recognizer sees**.

## References

1. N. Otsu, “A Threshold Selection Method from Gray-Level Histograms,” *IEEE Transactions on Systems, Man, and Cybernetics*, vol. 9, no. 1, pp. 62–66, 1979.
2. A. Rosenfeld and J. L. Pfaltz, “Sequential Operations in Digital Picture Processing,” *Journal of the ACM*, vol. 13, no. 4, pp. 471–494, 1966.
3. R. Smith, “An Overview of the Tesseract OCR Engine,” *Proceedings of the Ninth International Conference on Document Analysis and Recognition (ICDAR)*, 2007.
4. A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, “Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks,” *ICML*, 2006.
5. B. Shi, X. Bai, and C. Yao, “An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition,” *IEEE TPAMI*, vol. 39, no. 11, pp. 2298–2304, 2017.
