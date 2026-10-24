# Completed seven-class cacao ViT experiment

This package records actual training on the complete user-supplied archive of 7,428 images. It is a frozen pretrained MSN-Small Vision Transformer with a trained seven-class linear head. The backbone was not fine-tuned.

Weighted cross-entropy achieved mean accuracy 71.24% (sample SD 0.13 percentage points), macro F1 0.7015 (SD 0.0017), and macro ROC AUC 0.9480. Unweighted training achieved 70.43% accuracy and macro F1 0.6928. There are three classifier seeds per loss condition. The fitted model is not an ensemble. Paired bootstrap intervals for the seed-42 weighting difference included zero.

The fixed partition contains 5,566 training, 1,118 validation and 744 test images. All exact duplicate copies and ten screened near-copy links were grouped before splitting. There are 6,656 content groups with no group overlap across partitions. These are computational content groups, not independently verified physical bean identities. The archive contains 762 exact duplicate pairs in the unfermented class. The primary scoring retains image copies within their partition; a separate exact-hash test sensitivity analysis is archived.

## Package layout

- `cacao_vit/experiment.py`: preparation, metrics and the earlier generic full-fine-tuning recipe. That full-fine-tuning recipe was not used for the reported experiment.
- `cacao_vit/frozen_experiment.py`: the actual feature-extraction and classifier-training implementation.
- `cacao_vit/predict_frozen.py`: reconstructs a selected trained classifier for image prediction.
- `cacao_vit/extract_archive.py`: normalizes the supplied ZIP's class folder names safely.
- `complete_prepared/`: the exact manifest, counts and group-safe partition.
- `complete_audit/`: decoded-image audit, duplicate records, similarity audit and content-group assignments.
- `actual_frozen_msn_small/`: cached features, pinned checkpoint configuration, six trained heads, histories, predictions, metrics, aggregation and figures.
- `empirical_manuscript.md`: the updated one-column IMRAD manuscript source with 17 genuine references.
- Analysis and verification scripts at the package root reproduce the reported diagnostics.

The full image archive and pretrained backbone weights are not embedded. The user-supplied ZIP supplies the images. The encoder is obtained from the official Facebook MSN-Small Hugging Face repository at pinned revision `a50267de5e56f5b66a7fec6d2d8a8c60db24a704`. Only the six newly trained small classifier checkpoints are bundled.

## Installation

Use Python 3.12 in a virtual environment. For the verified CPU setup:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r cacao_vit/requirements.txt
```

The actual environment was Python 3.12.14, Linux, AMD EPYC 9V74 CPU, four PyTorch CPU threads, FP32 and no GPU. Exact package versions and checkpoint metadata are in `actual_frozen_msn_small/feature_configuration.json`.

## Extract and verify the original images

From the package root, extract the supplied archive into a new directory:

```bash
python cacao_vit/extract_archive.py --archive /path/to/dataset-20261012T120149Z-1-001.zip --out complete_cacao
```

The archive SHA-256 must be `1ffd8e383890dc7e2c41dbeb955130cba51eccdedec0e9d1599668716af8e165` to match this study. Training verifies decoded-pixel hashes against the fixed manifest. Do not silently substitute a different archive or regenerate a different split.

## Reproduce classifier fitting from the cached features

Create a new output directory and copy only `features.npz` and `feature_configuration.json` from `actual_frozen_msn_small` into it. Do not copy the completed run subdirectories if you intend to fit new heads. Then run:

```bash
python cacao_vit/frozen_experiment.py --root complete_cacao --manifest complete_prepared/manifest.csv --out rerun_frozen
```

The script verifies the images and cache manifest, then fits six new heads using the same recipe: weighted/unweighted cross-entropy; seeds 42, 52 and 62; AdamW learning rate 0.001 and weight decay 0.0001; batch size 64; 100 maximum epochs; patience 10; and validation macro-F1 checkpoint selection. There is no training augmentation, LR scheduler, gradient accumulation or mixed precision. No test labels are used in fitting or selection. Completed runs are skipped in an existing output directory, so a new directory is required for a fresh reproduction.

To re-extract the frozen representations, use a new output directory without the cache. Extraction uses deterministic aspect-preserving 224-pixel letterboxing and checkpoint-specific normalization. It took 369.79 seconds in the measured runtime. Seed variation measures classifier initialization and minibatch order, not encoder pretraining or alternative partitions.

## Predict with a trained checkpoint

```bash
python cacao_vit/predict_frozen.py --image complete_cacao/whole_beans/WB0001.png --classifier actual_frozen_msn_small/weighted_42/classifier.pt
```

The seed-42 weighted checkpoint is the fixed representative run. It is not selected for the highest test score. Direct reconstruction agreed with cached-feature predictions on seven actual images to maximum absolute probability error 1.79e-7.

## Results and uncertainty

Each run supplies its selected epoch, fitting history, head weights, 744 held-out predictions with seven probabilities, per-class metrics, confusion matrix and 2,000-resample content-group bootstrap intervals. `aggregate_results.json` reports mean and sample SD across seeds. The paired weighting comparison is in `analysis/paired_ablation.json` and the exact-hash test sensitivity analysis is in `analysis/exact_dedup_sensitivity.json`.

Average precision is calculated by scikit-learn and is not trapezoidal PR AUC. The Brier score sums squared errors across seven categories before averaging across images. ECE uses fifteen equal-width bins. Bootstrap intervals condition on fitted models and the present content groups. They are not confidence intervals for deployment across factories.

The dominant error is skin-damaged beans predicted as broken beans: 55 of 82 such test images in the representative run. Class weighting has a favorable small point difference, but the paired intervals include zero. The supplied labels were retained without independent biological reannotation. No CNN baseline, different encoder comparison, full-backbone fine-tuning or industrial deployment claim is supported by these runs.
