by Wenbo Hu (w1hu@ucsd.edu), Johnny Liu (jrl002@ucsd.edu)

---

## Introduction

Large-scale contrastive vision-language pre-training has shown significant progress in visual representation learning. Unlike traditional visual systems trained by a fixed set of discrete labels, a new paradigm was introduced in CLIP to directly learn to align images with raw texts in an open-vocabulary setting. On downstream tasks, a carefully chosen text prompt is employed to make zero-shot predictions. To avoid non-trivial prompt engineering, context optimization has been proposed to learn continuous vectors as task-specific prompts with few-shot training examples. An alternative path is fine-tuning with feature adapters on either the visual or language branch, which learns new features and performs residual style feature blending with the original pre-trained features and also showed good performance. However, one of CLIP's limitations is a counter-intuitive drop in performance when transitioning from a zero-shot to a few-shot setting for complex datasets is less well studied. In our approach, we are proposing a method to combine CLIP’s strong zero-shot performance with efficient few-shot learning. In our work, we propose to learn a good prompt using context optimization, use heavy data augmentation, and perform residual style feature blending with newly learned image features to achieve a better and consistent few-shot performance improvement.

---

## Methodology 


---

## Results


---

## Discussion


---

## Conclusion

---
