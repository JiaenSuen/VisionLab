# Literature Notes and Conceptual Connections

## Otsu Thresholding

Nobuyuki Otsu's 1979 method selects a threshold from a gray-level histogram by optimizing class separation. The relevance to this project is not only the threshold formula but the dependence on the input distribution. If unrelated image regions are included before threshold estimation, they change the histogram and can change the selected threshold. This provides a strong theoretical explanation for the observed ROI-order regression. Citation: N. Otsu, “A Threshold Selection Method from Gray-Level Histograms,” *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1):62–66, 1979. DOI: 10.1109/TSMC.1979.4310076.

## Connected Components

Connected-component labeling is a classical operation in digital image processing. Rosenfeld and Pfaltz described sequential neighborhood operations for labeling connected subsets in digitized pictures. In the present project, connected components are used as an interpretable proposal mechanism after binary segmentation. The historical connection is useful because it reinforces the correct interpretation: the algorithm labels topological connectivity, not semantic characters. Citation: A. Rosenfeld and J. L. Pfaltz, “Sequential Operations in Digital Picture Processing,” *Journal of the ACM*, 13(4):471–494, 1966. DOI: 10.1145/321356.321357.

## CTC and Unsegmented Recognition

Graves et al. introduced Connectionist Temporal Classification (CTC) for sequence labeling when explicit alignment between input positions and target labels is unavailable. This directly addresses the project's concern that exact character boundaries are unreliable. A sequence crop can be represented as a series of visual features while CTC learns an alignment to the target string. Citation: A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, “Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks,” *ICML*, 2006, pp. 369–376. DOI: 10.1145/1143844.1143891.

## CRNN for Image-Based Sequence Recognition

Shi, Bai, and Yao proposed an end-to-end CRNN architecture integrating convolutional feature extraction, sequence modeling, and transcription for image-based sequence recognition. A central property relevant here is that the model can handle variable-length sequences without requiring character-level segmentation. This supports the project's decision to group nearby components into a line crop rather than force every connected component into one character. Citation: B. Shi, X. Bai, and C. Yao, “An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition,” *IEEE TPAMI*, 39(11):2298–2304, 2017. DOI: 10.1109/TPAMI.2016.2646371.

## How the Literature Should Be Used in This Project

The literature does not imply that the current method is novel merely because it combines known operations. The research value currently lies in the problem-specific analysis: identifying how ROI scope changes adaptive threshold behavior, documenting the topology/recognition trade-off, and designing a lightweight transition from chromatic segmentation to sequence OCR. A stronger publication-oriented contribution would require a labeled dataset, systematic ablations, quantitative gains, and possibly a new adaptive grouping or scale-normalization method that is compared against established OCR baselines.
