# CRNN Modularization and Transfer-Learning Analysis

## Origin of the Model

A previously trained CRNN project was integrated into the red-text OCR repository. The historical network combines two convolutional stages, a projection layer, a two-layer bidirectional GRU, and a CTC-compatible output classifier. Its checkpoint contains 36 alphanumeric classes (`0-9A-Z`) plus the CTC blank class. The model was originally trained on CAPTCHA imagery resized to 75 x 300 pixels and achieved strong performance in that source task according to the archived training project.

## Refactoring

The original implementation was not directly reusable as a package because model code imported a global `DEVICE` value from utilities while the utilities also imported the model. It also assumed CUDA and mixed model construction, image transforms, decoding, training helpers, and visualization in shared modules. The integration separates these responsibilities into `modules/crnn/model.py`, `codec.py`, `preprocessing.py`, and `inference.py`. The neural layer shapes were intentionally preserved so the historical checkpoint loads unchanged. Device selection is now runtime-safe, and inference can be called on a NumPy/OpenCV crop without depending on the former training project.

## Transfer Smoke Test

The checkpoint successfully loaded and executed on all grouped document crops, so the software integration is verified. Recognition, however, did not transfer. Under the historical stretch resize, most sequences were decoded as empty strings because the CTC blank class dominated nearly every timestep; one small group produced a single `8`. Under aspect-ratio-preserving letterbox preprocessing, all tested groups were blank-dominated. This result should not be interpreted as a failure of CRNN as an architecture. It is evidence that the old model's learned visual distribution is substantially different from the new document domain.

The differences include foreground color, font family, stroke scale, document security texture, character spacing, the number of characters per sequence, and vocabulary. The historical model also cannot represent lowercase letters, hyphens, or other punctuation. Therefore, even perfect localization cannot yield correct strings outside `0-9A-Z` with this output layer.

## Research Consequence

The old checkpoint is retained as a reproducible transfer baseline, not a final recognizer. The next model experiment should first define the target character inventory, then construct real or synthetic training crops that use the exact grouping/crop protocol produced by this repository. A useful comparison would be frozen historical CRNN, fine-tuned CRNN, and newly trained CRNN under the same evaluation split. CER and exact sequence accuracy should be reported together with the localization/grouping metrics so model failure is not confused with segmentation failure.
