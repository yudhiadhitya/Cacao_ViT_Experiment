# Seven Class Indonesian Cocoa Bean Classification Using Frozen Vision Transformer Representations

## Abstract

Image-based cocoa-bean sorting requires reliable discrimination among categories with overlapping visual attributes. This study evaluates a frozen pretrained Vision Transformer on the seven-class Indonesian cocoa-bean collection supplied as a complete archive of 7,428 images. An integrity audit identified 762 exact duplicate pairs and ten additional near-copy links. Related content was grouped before dividing the collection into 5,566 training, 1,118 validation and 744 test images. A masked-siamese-network-pretrained ViT-Small encoder produced 384-dimensional representations, and seven-class linear classifiers were trained with weighted or unweighted cross-entropy under three random seeds. Checkpoints were selected using validation macro F1. Weighted training achieved mean test accuracy of 71.24 ± 0.13% and macro F1 of 0.7015 ± 0.0017, compared with 70.43 ± 0.23% and 0.6928 ± 0.0016 for unweighted training. Macro ROC AUC was 0.9480. Class-wise evaluation exposed substantial confusion between skin-damaged and broken beans despite perfect classification of the 40 bean-fraction test images in the representative run. Paired content-group bootstrap intervals for the weighting benefit included zero. A sensitivity analysis retaining one prediction per exact image hash reduced representative accuracy to 69.67%, while macro F1 remained 0.7016. These findings support frozen ViT representations as a reproducible transfer baseline, while identifying dataset redundancy and class ambiguity as important evaluation constraints. The experiment trains the classification head and does not establish full-backbone fine-tuning performance or generalization to independent factories.

Keywords: cocoa bean classification; Vision Transformer; masked siamese networks; linear probing; class imbalance; duplicate-aware evaluation; transfer learning.

## 1 Introduction

Automated cocoa-bean classification can support processing decisions when image labels reflect relevant differences in bean condition. The technical problem involves subtle distinctions within one commodity rather than recognizing unrelated objects. Adhitya et al. introduced an Indonesian cocoa-image classification study with seven categories and conventional feature-based comparisons [1]. Later work on the same collection investigated contextual feature adaptation [2]. These studies provide an application setting in which transferred visual representations can be evaluated, but they do not determine whether a pretrained transformer will separate every category reliably.

Texture methods summarize local pixel relationships, as exemplified by gray-level co-occurrence representations [3]. Agricultural machine-learning applications also employ learned features across multiple sensing modalities and decision tasks [4]. Their relevance to cocoa sorting depends on the target images, the annotation scheme and the evaluation partition. Differences in preprocessing or repeated observations can affect apparent performance even when the architecture and optimization code are valid.

Transformers model relationships among sequence elements through attention [5]. Vision Transformers apply this approach to image patches and transfer representations learned from large image collections to downstream recognition [6]. DeiT established that transformer training can be made more data efficient through a particular training and distillation design [7]. Masked Siamese Networks, or MSN, use self-supervised pretraining based on matching representations of masked and unmasked views [8]. This study uses an existing MSN-Small checkpoint as a frozen encoder, rather than training the MSN pretraining objective on cocoa images.

A frozen-encoder experiment answers a narrower question than full fine-tuning. It tests whether fixed pretrained representations support a useful downstream decision boundary. It also provides a practical baseline when accelerator resources are unavailable: the images pass through the encoder once, after which a small classifier can be optimized repeatedly. This design preserves a clear distinction between transferred feature quality and adaptation of the encoder itself. A residual CNN remains a relevant alternative [9], but a matched CNN comparison is outside the completed experiment.

The evaluation must also account for class imbalance and related image content. Overall accuracy can conceal errors in smaller or harder categories, and duplicate copies crossing partition boundaries can create leakage. The broader reproducibility literature demonstrates why data handling is part of the scientific claim, rather than merely an implementation detail [15]. A supplied archive makes actual experimentation possible, but does not automatically establish independent physical beans or acquisition batches.

The objectives are to establish an auditable seven-class transfer baseline; compare weighted and unweighted classifier training under fixed representations and partitions; characterize class-wise errors and probability quality; and measure how duplicate-aware handling affects interpretation. The contributions are the completed experiment, explicit content-group partition, archived run outputs and a class-sensitive analysis. No new transformer architecture or state-of-the-art performance claim is proposed.

## 2 Methods

### Dataset and archive verification

The complete dataset was supplied as a ZIP archive containing 7,428 PNG images and two excluded nonimage entries. Its integrity check passed, and its class counts matched those reported by Adhitya et al. [1]. The source publication identifies a cocoa-processing setting in South Sulawesi, Indonesia. The uploaded files were normalized into seven class folders without changing their image bytes. The actual dimensions ranged from 12 to 3,360 pixels in width and from 26 to 2,100 pixels in height. Counts and realized partitions are reported in Table 1.

Table 1. Verified archive counts and the fixed content-group partition. All source images were retained.

| Category | Full archive | Training | Validation | Test |
|---|---|---|---|---|
| Whole beans | 1187 | 895 | 174 | 118 |
| Broken beans | 1046 | 791 | 148 | 107 |
| Bean fractions | 426 | 315 | 71 | 40 |
| Skin-damaged beans | 822 | 615 | 125 | 82 |
| Fermented beans | 916 | 692 | 136 | 88 |
| Unfermented beans | 1776 | 1318 | 276 | 182 |
| Moldy beans | 1255 | 940 | 188 | 127 |
| Total | 7428 | 5566 | 1118 | 744 |


The folder names provide the labels used in this experiment. No new expert annotations, chemical measurements or physical bean identifiers were supplied. The single-label mapping was retained without imposing a new assignment precedence. This distinction matters because morphological and fermentation attributes may coexist in one bean. The experiment evaluates recognition of the supplied labels; it does not independently validate the biological or commercial interpretation of those labels.

### Duplicate audit and partition construction

Images were decoded after applying EXIF orientation, converted to RGB and hashed using their dimensions and pixel bytes. The audit found 6,666 unique decoded image hashes. All 762 exact duplicate pairs occurred within the unfermented category, and no identical decoded image had conflicting labels. The 1,524 files participating in these pairs accounted for 20.52% of the archive. Their presence required a grouping rule to prevent copies of the same content from appearing in both training and testing.

A second screen used a 63-bit perceptual hash derived from the low-frequency discrete cosine coefficients of a 32-pixel grayscale image, excluding the DC coefficient. Pairs at Hamming distance four or less were compared after trimming near-black background pixels and resizing their RGB content to 64 by 64 pixels. A link required correlation of at least 0.98 and root-mean-square error no greater than 0.02 on intensities scaled to zero through one. This screen generated 316 candidate pairs and accepted ten links, which were visually inspected. One accepted link connected unfermented and whole-bean labels. Labels were preserved, and linked images were assigned to the same content group.

Joining exact duplicate pairs and accepted near-copy links yielded 6,656 content groups. These are computational similarity units, not verified physical bean identities. A ten-fold stratified group partition, using seed 42, supplied the test fold. A six-fold stratified group partition of the remainder supplied validation. The realized split contained 5,566 training images in 4,993 groups, 1,118 validation images in 998 groups, and 744 test images in 665 groups. No content group overlapped partitions. Grouping can alter exact proportions, so the achieved counts are reported instead of rounded target percentages.

The image-level partition supports evaluation on withheld content from the supplied collection. It cannot exclude all repeated physical beans or establish generalization across factories, cameras or batches. Exact image hashes were verified against the manifest before feature extraction. The complete manifest, content-group assignments and audit thresholds are preserved in the experiment package.

### Frozen Vision Transformer representations

The encoder was facebook/vit-msn-small at repository revision a50267de5e56f5b66a7fec6d2d8a8c60db24a704. The network uses twelve transformer blocks, six attention heads per block, embedding width 384 and a feed-forward width of 1,536. Each 224 by 224 input is represented by 196 patches of 16 by 16 pixels plus a classification token. The final classification-token representation supplies the 384-dimensional feature vector [6,8].

Input images were converted to RGB and scaled to fit 224 by 224 pixels while preserving aspect ratio, using bicubic interpolation and neutral-gray padding. ImageNet channel means of 0.485, 0.456 and 0.406 and standard deviations of 0.229, 0.224 and 0.225 were applied. No training augmentation was used in this linear-probe experiment. The encoder remained in evaluation mode with all parameters frozen, and no data-dependent transformation was fitted on the test set.

Representations were extracted deterministically for all images and cached. Extracting a test representation with an unchanged pretrained encoder does not use test labels to fit or select the classifier. Classifier optimization accessed only training features, while validation features supported checkpoint selection. The encoder contains 21,665,664 parameters. Adding the seven-output linear head gives 21,668,359 total parameters, of which only 2,695 were trained. This is an actual frozen-representation experiment and does not constitute full-backbone fine-tuning.

### Classifier training and loss comparison

A linear layer mapped the 384-dimensional representation to seven logits, followed by softmax for prediction. The primary condition used inverse-frequency weighted cross-entropy, with each weight calculated as the training-image count divided by seven times that class's training count. The comparison condition used unweighted cross-entropy. All loss weights were calculated from training labels only.

Both conditions used AdamW [10] with learning rate 0.001, weight decay 0.0001, batch size 64, a maximum of 100 epochs and early-stopping patience of ten epochs. The learning rate was constant, and decay applied to the head parameters. There was no scheduler, gradient accumulation or mixed precision in this experiment. These settings were fixed before examining test predictions. Independent head initializations and minibatch shuffles used seeds 42, 52 and 62 on the same feature cache and partition.

Checkpoint selection maximized validation macro F1, retaining the earlier checkpoint on a tie. Each selected head was evaluated once on the test images. The six completed runs represent two loss conditions under three classifier seeds. They do not represent six independently pretrained encoders or an ensemble. A deterministic baseline selected the most frequent training label and predicted that label for every test image.

### Evaluation and uncertainty

Macro F1 was the primary outcome, calculated by averaging the class-specific harmonic means of precision and recall. Overall accuracy, balanced accuracy, weighted F1 and per-class precision and recall were exported. Balanced accuracy is the unweighted mean of per-class recall. Fixed labels and class order were used in all manifests, prediction files and confusion matrices.

Probability evaluation used macro one-versus-rest ROC AUC and macro average precision. The precision-recall perspective is useful under class imbalance [11]. Average precision was calculated by scikit-learn and was not relabeled as trapezoidal precision-recall AUC. Log loss, the multiclass Brier score and expected calibration error, or ECE, described probability quality [12]. ECE used fifteen equal-width top-confidence bins. The Brier score sums squared probability errors across classes and then averages across images. No calibration parameters were fitted.

Results were summarized as means and sample standard deviations across the three classifier seeds. For a fixed representative run, seed 42, 2,000 percentile bootstrap resamples of the 665 test content groups supplied 95% intervals for accuracy and macro F1. A paired group bootstrap compared weighted and unweighted heads at the same seed and on identical test images. These intervals condition on fitted heads and the present collection; they do not measure cross-factory or encoder-pretraining uncertainty.

A secondary sensitivity analysis retained one prediction per exact decoded pixel hash in the test set. It did not retrain the classifier or collapse distinct near-copy hashes. Seed 42 was used for representative figures and class-wise analysis because it was specified in advance, rather than selected for the highest test score. Numerical verification reconstructed the encoder plus saved head and compared direct image predictions with the cached-feature predictions.

### Implementation and compute environment

The implementation used PyTorch [16] and Hugging Face Transformers [17] with Python 3.12.14, PyTorch 2.8.0+cpu, torchvision 0.23.0+cpu, Transformers 4.57.1, huggingface-hub 0.35.3, NumPy 2.2.6, SciPy 1.16.2, scikit-learn 1.7.2, Pillow 11.3.0 and Matplotlib 3.10.7. Execution used an AMD EPYC 9V74 CPU under Linux, four PyTorch CPU threads, FP32 and no GPU. Image loading used zero worker subprocesses, and extraction batch size was sixteen. This reports the hardware actually used, rather than a proposed workstation.

The pretrained checkpoint revision, archive hash, manifest hash, classifier weights, histories and prediction probabilities were preserved. The source archive SHA-256 is 1ffd8e383890dc7e2c41dbeb955130cba51eccdedec0e9d1599668716af8e165. Direct reconstruction of the trained seed-42 weighted model agreed with cached-feature predictions on seven actual images, with maximum absolute probability difference of 1.79e-07.

## 3 Results

### Training and validation selection

All six classifier runs completed. Weighted checkpoints were selected at epochs 34, 24 and 35, and fitting stopped at epochs 44, 34 and 45 for seeds 42, 52 and 62. Unweighted checkpoints were selected at epochs 40, 15 and 11, with stopping at epochs 50, 25 and 21. The maximum epoch limit was therefore not reached. Table 2 records the actual selected checkpoints and test results. Figure 1 shows the representative fitting histories.

Table 2. Actual run-level results. Validation macro F1 selected the checkpoint before test evaluation.

| Loss | Seed | Selected epoch | Validation macro F1 | Test accuracy percent | Test macro F1 |
|---|---|---|---|---|---|
| Weighted | 42 | 34 | 0.6931 | 71.37 | 0.7026 |
| Weighted | 52 | 24 | 0.6840 | 71.24 | 0.6995 |
| Weighted | 62 | 35 | 0.6832 | 71.10 | 0.7025 |
| Unweighted | 42 | 40 | 0.6913 | 70.56 | 0.6921 |
| Unweighted | 52 | 15 | 0.6701 | 70.16 | 0.6917 |
| Unweighted | 62 | 11 | 0.6696 | 70.56 | 0.6946 |


[FIGURE_LEARNING]

Figure 1. Actual training loss and validation macro F1 for the weighted and unweighted seed-42 runs. The loss curves correspond to different weighting objectives and should not be interpreted as numerically identical optimization targets.

Feature extraction required 369.79 seconds for the full archive. Weighted classifier fitting and validation took 2.83 ± 0.34 seconds per run after caching, and unweighted fitting took 2.32 ± 1.27 seconds. These timings exclude neither validation computation nor checkpoint selection, but they exclude the subsequent bootstrap analysis. They are not an inference-latency benchmark.

### Overall held-out performance

Weighted training achieved mean accuracy of 71.24 ± 0.13% and macro F1 of 0.7015 ± 0.0017. Its balanced accuracy was 70.26 ± 0.20%. Unweighted training achieved 70.43 ± 0.23% accuracy and macro F1 of 0.6928 ± 0.0016. Table 3 summarizes the comparison. The training-majority baseline predicted unfermented beans and achieved 24.46% accuracy, macro F1 of 0.0562 and balanced accuracy of 14.29%.

Table 3. Test metrics as mean ± sample standard deviation across three classifier seeds. The deterministic majority baseline has no seed variation.

| Condition | Accuracy percent | Macro F1 | Balanced accuracy percent |
|---|---|---|---|
| Weighted loss | 71.24 ± 0.13 | 0.7015 ± 0.0017 | 70.26 ± 0.20 |
| Unweighted loss | 70.43 ± 0.23 | 0.6928 ± 0.0016 | 69.33 ± 0.27 |
| Training-majority baseline | 24.46 | 0.0562 | 14.29 |


In the representative weighted run, 531 of 744 test images were correctly classified. Its accuracy was 71.37%, with a group-bootstrap interval of 67.88% to 74.83%. Macro F1 was 0.7026, with interval 0.6722 to 0.7317. The mean weighting benefit was 0.81 percentage points of accuracy and 0.0087 macro F1. These small differences require the uncertainty analysis rather than a superiority conclusion based on the point estimates alone.

### Class-wise errors

Table 4 shows class-wise performance for the representative weighted run. All 40 bean-fraction test images were correctly classified, with no false-positive fraction predictions. This is a result on forty withheld images and does not establish perfect recognition outside the present collection. Moldy beans achieved F1 of 0.9306, and unfermented beans achieved 0.8691. Skin-damaged beans had F1 of only 0.3243, substantially below the other categories.

Table 4. Class-wise test results for the weighted seed-42 checkpoint. Support counts include retained duplicate copies within their assigned test groups.

| Category | Test images | Precision | Recall | F1 |
|---|---|---|---|---|
| Whole beans | 118 | 0.6410 | 0.6356 | 0.6383 |
| Broken beans | 107 | 0.5299 | 0.6636 | 0.5892 |
| Bean fractions | 40 | 1.0000 | 1.0000 | 1.0000 |
| Skin-damaged beans | 82 | 0.3636 | 0.2927 | 0.3243 |
| Fermented beans | 88 | 0.5543 | 0.5795 | 0.5667 |
| Unfermented beans | 182 | 0.8814 | 0.8571 | 0.8691 |
| Moldy beans | 127 | 0.9661 | 0.8976 | 0.9306 |


[FIGURE_CONFUSION]

Figure 2. Observed confusion matrix for weighted seed 42. Rows are true categories and columns are predicted categories. Counts sum to the 744 test images.

The largest confusion was skin-damaged beans predicted as broken beans: 55 of 82 skin-damaged images followed this error path. The reverse error affected 31 of 107 broken-bean images. Whole and fermented categories were also confused, with 28 whole beans predicted as fermented and 24 fermented beans predicted as whole. These patterns indicate that the model's overall score hides uneven category discrimination. They are consistent with visual overlap in the supplied labels, but the experiment does not establish whether annotation ambiguity or representation limitations caused each error.

### Ranking and probability quality

Weighted training produced macro ROC AUC of 0.9480 ± 0.0007 and macro average precision of 0.7388 ± 0.0022. The corresponding unweighted values were 0.9484 ± 0.0010 and 0.7389 ± 0.0042. Weighted log loss averaged 0.6479 ± 0.0049, and ECE averaged 0.0461 ± 0.0087. Table 5 reports the probability metrics, and Figure 3 presents the representative ROC and precision-recall curves.

Table 5. Probability metrics as mean ± sample standard deviation across seeds. ECE uses fifteen bins, and lower log loss and ECE indicate better probability quality under their definitions.

| Loss | Macro ROC AUC | Macro average precision | Log loss | ECE |
|---|---|---|---|---|
| Weighted | 0.9480 ± 0.0007 | 0.7388 ± 0.0022 | 0.6479 ± 0.0049 | 0.0461 ± 0.0087 |
| Unweighted | 0.9484 ± 0.0010 | 0.7389 ± 0.0042 | 0.6599 ± 0.0304 | 0.0664 ± 0.0219 |


[FIGURE_ROCPR]

Figure 3. Observed one-versus-rest ROC and precision-recall curves for the weighted seed-42 classifier. Curves use actual saved test probabilities; no illustrative or simulated scores are included.

Macro ROC AUC was similar between loss conditions even though their hard-label macro-F1 scores differed. The ROC summary describes one-versus-rest ranking across thresholds, whereas the reported F1 uses the seven-class argmax decision. High ranking performance therefore did not eliminate the broken-versus-skin-damaged decision errors. Calibration metrics were reported without temperature fitting, and the modest ECE values should not be interpreted as proof that every class is equally calibrated.

### Paired ablation and duplicate sensitivity

For seed 42, weighting increased accuracy by 0.81 percentage points and macro F1 by 0.0105. The paired content-group bootstrap interval was -1.46 to 2.84 percentage points for accuracy and -0.0095 to 0.0289 for macro F1. Both intervals included zero. The completed ablation therefore shows a small favorable point difference under this split, with insufficient evidence to assert a reliable population-level improvement.

Removing repeated exact hashes from the test summary retained 666 predictions. Representative accuracy decreased from 71.37% to 69.67%, while macro F1 changed from 0.7026 to 0.7016. Duplicate copies thus affected the image-weighted accuracy more than the macro-F1 point estimate in this comparison. This sensitivity analysis changes the evaluation weighting without retraining the model and does not resolve unknown physical bean identities.

## 4 Discussion

### Interpretation of the actual experiment

The completed experiment demonstrates that a fixed MSN-Small representation supports meaningful seven-class discrimination on withheld content from the supplied Indonesian cocoa collection. Accuracy substantially exceeded the training-majority baseline, and the trained head required only 2,695 optimized parameters. This establishes a concrete transfer-learning baseline under the reported partition. It does not establish that attention itself causes the performance advantage, because the study did not remove attention or compare matched alternative encoders.

Class weighting produced consistent favorable macro-F1 point estimates across the three seeds, but the paired test-sampling intervals included zero. The result should therefore be interpreted as an observed small difference, rather than a confirmed improvement applicable to other samples or factories. The classifier-seed standard deviations were small relative to the group-bootstrap interval widths. Initialization stability on one feature cache does not substitute for uncertainty about new data.

The strongest category results were not representative of all labels. Bean fractions were perfectly classified in the small held-out subset, whereas skin-damaged beans had poor recall and F1. Both aggregate accuracy and macro ROC AUC would conceal part of this contrast if reported alone. A sorting application should examine category-specific costs and review pathways rather than assume that a single accuracy number establishes acceptable behavior for every class.

### Dataset integrity and the meaning of independence

The duplicate audit identified substantial redundancy in the unfermented category and additional closely related content elsewhere. Grouping prevented the detected copies from crossing training, validation and test boundaries. That constraint is stronger than an unrestricted file-level random split. It still relies on measurable content similarity and cannot verify that every group corresponds to one independent physical bean.

The accepted link between whole and unfermented labels also illustrates why the category taxonomy requires attention. A physical bean can be intact and have a fermentation status, while the dataset supplies a single folder label. The experiment retained the supplied labels and did not infer a new labeling rule. Expert review of overlapping attributes and independent acquisition metadata would be needed to distinguish annotation ambiguity from model error more conclusively.

The reduced accuracy after exact-hash deduplication shows that scoring every image file and scoring distinct decoded content need not yield the same result. Reporting both summaries makes that dependence visible. It would be inappropriate to present the small training-seed variation as a guarantee of broader reproducibility while omitting the archive's redundancy or the unavailable physical identifiers.

### Relationship to previous cocoa-image work

Adhitya et al. reported 65.08% accuracy for their GLCM-XGBoost configuration [1], and SCANet reported 88.72% for its contextual approach [2]. The present frozen-ViT result lies between these historical accuracy values, but the partitions, duplicate handling and training protocols are not shown to match. Consequently, the numbers do not support a controlled ranking among those systems. Reproducing the historical baselines on the present manifest would be necessary for a defensible comparative claim.

The present study evaluates one frozen pretrained encoder, not the relative merits of supervised ViT-Base, MSN-Base, CNNs and MSN-Small. Its practical advantage is that one encoder pass supports several inexpensive classifier runs. A fuller study could investigate backbone fine-tuning or an independently acquired test collection, but such outcomes are not represented by the present measurements. The successful CPU experiment should not be described as an accelerator benchmark or an industrial throughput test.

Attention maps can provide exploratory descriptions of token interactions [13], but visually plausible maps are not automatically faithful explanations of prediction mechanisms [14]. No causal localization or expert-validated defect segmentation is claimed here. The conclusions follow from actual predictions, class-wise errors and the documented evaluation design rather than from a selected attention visualization.

### Limitations

The experiment uses one fixed partition and three classifier seeds. Repeated grouped partitions or a new factory-level test set were not evaluated. Similarity grouping may miss transformed copies or distinct views of the same bean. The source archive lacks independently verified physical identifiers, acquisition batches and annotation precedence. These restrictions limit the generalization claim to the supplied image collection under the reported protocol.

The frozen encoder was not adapted to cocoa images, and deterministic preprocessing omitted training augmentation. This makes the experiment a linear-probe baseline rather than an optimized endpoint for the task. The perfect fraction result involves only forty test images. Duplicate copies retained within test groups affect image-weighted summaries, and the secondary distinct-hash analysis does not provide a separate retrained benchmark. Finally, recognition of a moldy-image label does not establish chemical contamination or compliance with a commercial grading standard.

## 5 Conclusion

Actual training and evaluation on the complete 7,428-image cacao archive produced a reproducible seven-class frozen-ViT baseline. After content grouping, the test set contained 744 images, and weighted classifier training achieved 71.24% mean accuracy and macro F1 of 0.7015 across three seeds. The matched unweighted condition achieved 70.43% and 0.6928. Small weighting benefits were not resolved by paired bootstrap uncertainty. The main practical findings are the viability of transferred representations, severe broken-versus-skin-damaged confusion, and the effect of dataset redundancy on interpretation. These results concern a frozen encoder and a trained linear head; independent acquisition data and matched baselines remain necessary for broader claims.

## Data and code availability

The supplied archive is identified by its filename, dataset counts and SHA-256 in the experiment records. Dataset provenance is linked to the source publication [1] and the user-provided Google Drive folder, https://drive.google.com/drive/folders/1x2kHr23IxhpSryvce1UilBEI3pv5k5BQ . The uploaded archive enabled complete access despite the earlier download interruption. No new redistribution license is asserted.

The reproducibility package contains the fixed manifest, content-group audit, pinned encoder revision, cached representations, six trained classifier checkpoints, optimization histories, test probabilities, metrics, bootstrap summaries, scientific figures and reconstruction verification. The package supports rerunning classifier training from the cache or re-extracting features from the original archive. It does not contain a newly fine-tuned encoder or fabricated cacao results.

## References


[1] Adhitya, Y.; Prakosa, S. W.; Köppen, M.; Leu, J.-S. Feature Extraction for Cocoa Bean Digital Image Classification Prediction for Smart Farming Application. Agronomy 2020, 10(11), 1642. https://doi.org/10.3390/agronomy10111642

[2] Sigalingging, X.; Prakosa, S. W.; Leu, J.-S.; Hsieh, H.-Y.; Avian, C.; Faisal, M. SCANet: Implementation of Selective Context Adaptation Network in Smart Farming Applications. Sensors 2023, 23(3), 1358. https://doi.org/10.3390/s23031358

[3] Haralick, R. M.; Shanmugam, K.; Dinstein, I. Textural Features for Image Classification. IEEE Transactions on Systems, Man, and Cybernetics 1973, SMC-3(6), 610-621. https://doi.org/10.1109/TSMC.1973.4309314

[4] Liakos, K. G.; Busato, P.; Moshou, D.; Pearson, S.; Bochtis, D. Machine Learning in Agriculture: A Review. Sensors 2018, 18(8), 2674. https://doi.org/10.3390/s18082674

[5] Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; Polosukhin, I. Attention Is All You Need. Advances in Neural Information Processing Systems 2017, 30. https://arxiv.org/abs/1706.03762

[6] Dosovitskiy, A.; et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. International Conference on Learning Representations 2021. https://arxiv.org/abs/2010.11929

[7] Touvron, H.; Cord, M.; Douze, M.; Massa, F.; Sablayrolles, A.; Jégou, H. Training Data-Efficient Image Transformers and Distillation through Attention. Proceedings of Machine Learning Research 2021, 139, 10347-10357. https://proceedings.mlr.press/v139/touvron21a.html

[8] Assran, M.; Caron, M.; Misra, I.; Bojanowski, P.; Bordes, F.; Vincent, P.; Joulin, A.; Rabbat, M.; Ballas, N. Masked Siamese Networks for Label-Efficient Learning. arXiv preprint 2022, arXiv:2204.07141. https://doi.org/10.48550/arXiv.2204.07141

[9] He, K.; Zhang, X.; Ren, S.; Sun, J. Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 2016, 770-778. https://doi.org/10.1109/CVPR.2016.90

[10] Loshchilov, I.; Hutter, F. Decoupled Weight Decay Regularization. International Conference on Learning Representations 2019. https://arxiv.org/abs/1711.05101

[11] Saito, T.; Rehmsmeier, M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE 2015, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

[12] Guo, C.; Pleiss, G.; Sun, Y.; Weinberger, K. Q. On Calibration of Modern Neural Networks. Proceedings of Machine Learning Research 2017, 70, 1321-1330. https://proceedings.mlr.press/v70/guo17a.html

[13] Abnar, S.; Zuidema, W. Quantifying Attention Flow in Transformers. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics 2020, 4190-4197. https://doi.org/10.18653/v1/2020.acl-main.385

[14] Adebayo, J.; Gilmer, J.; Muelly, M.; Goodfellow, I.; Hardt, M.; Kim, B. Sanity Checks for Saliency Maps. Advances in Neural Information Processing Systems 2018, 31. https://arxiv.org/abs/1810.03292

[15] Kapoor, S.; Narayanan, A. Leakage and the Reproducibility Crisis in Machine-Learning-Based Science. Patterns 2023, 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804

[16] Paszke, A.; et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems 2019, 32. https://arxiv.org/abs/1912.01703

[17] Wolf, T.; et al. Transformers: State-of-the-Art Natural Language Processing. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing System Demonstrations 2020, 38-45. https://doi.org/10.18653/v1/2020.emnlp-demos.6
