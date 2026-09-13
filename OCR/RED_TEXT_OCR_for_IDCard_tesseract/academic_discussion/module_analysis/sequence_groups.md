# Module Analysis: `modules/sequence_groups.py`

This module converts connected-component proposals into sequence-level text regions. It deliberately does not perform OCR. The separation is important because grouping quality can be inspected independently from recognizer performance.

Boxes are connected when they have compatible vertical alignment and sufficiently small horizontal distance. Union-find implements transitive merging, so a sequence can grow through a chain of neighboring proposals. The optional adaptive gap uses median component height as a scale proxy, avoiding the assumption that one fixed pixel distance is valid across resolutions.

The module also separates localization boxes from recognition crops. `expand_boxes_for_ocr()` adds fixed or height-adaptive margins within image boundaries. `export_group_crops()` samples those expanded regions from the original image rather than the binary mask. This preserves anti-aliased edges and surrounding context that may be valuable to Tesseract or a future learned recognizer.
