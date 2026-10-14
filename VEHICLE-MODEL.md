# Dominant vehicle evidence

The optional car fallback uses Megvii YOLOX-Nano (COCO, release `0.1.1rc0`):
https://github.com/Megvii-BaseDetection/YOLOX

Copyright (c) Megvii, Inc. and its affiliates. Apache-2.0 license text is in
`VEHICLE-MODEL-LICENSE.txt`. The Python adapter here is a new implementation;
it uses Pillow bilinear BGR preparation rather than claiming exact OpenCV
preprocessing equivalence. Both development and validation used that adapter.

The fixed official HTTPS model URL and SHA-256 are in `vehicle_detector.py`.
On first need, the server downloads the model (maximum 5 MiB, checksum checked)
and caches it on its ephemeral disk. Download failure preserves the previous
uncertain category decision, and is retried no more than once per five minutes.
The HTTP stream uses connect/read timeouts and a transfer time check. No image
or user metadata is included in the model download. The initial fallback may
be slower after a new server deployment.

Only a previously uncertain **car** category with property score below 0.2 can
use this fallback. A COCO car box must score at least 0.5, occupy at least 15%
of the image, and have its center within the central 70% in both axes. Explicit
category mismatches are preserved. The property and general modes do not use
the detector, and the underlying AI-generation score and thresholds are unchanged.

14 October 2026 local research checks:
- Reused development car set: category acceptance 151/200 → 159/200; end-to-end
  correct outcomes 130 → 136, wrong outcomes 19 → 21, uncertain 2 → 2, rejected
  49 → 41. More accepted images can also expose errors in the unchanged AI model.
- Frozen separate Stanford Cars subset: acceptance 48/50 → 50/50. This measures
  category recognition only, not AI/real accuracy. Publisher vehicle classes,
  not independently verified photographic provenance; model-training overlap
  and perceptual duplicates cannot be ruled out.
- Separate 100 LSUN bedrooms and 50 cat/dog examples: no car acceptances before
  or after. These are small research sets, not a general safety guarantee.
- Caption-selected Defactify scenes were inspected and found to contain trains,
  toys and small background vehicles. They are excluded from category success
  claims. AI labels in that corpus are not car-category annotations.
- A property full-frame preprocessing candidate was rejected: development
  151/200 → 162/200 correct, but separate 100-bedroom validation 73 → 75 correct,
  unchanged 25 wrong and REAL-to-AI errors 8 → 9. It failed the predeclared gate.
  Property preprocessing is therefore unchanged.

Manifests, frozen policies, raw outcomes and limitations are kept in the local
evaluation artifacts. There is no verified real-listing, fraud, damage or
ownership accuracy claim. No research images are bundled into the deployment.
