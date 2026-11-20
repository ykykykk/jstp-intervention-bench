# Joint State Transition Prediction: Goal Generalization with Sparse Action Labels

Ke Yue · Independent Researcher · v1.2 · Protocol clarified 21 November 2026

## Abstract

A controller can learn what an action does without being told when to take it. Does that knowledge help when a new goal arrives? Joint State Transition Prediction (JSTP) treats action and environmental change as components of the same transition; it does not require a full reconstruction of the future. We varied the number of action labels in a fixed bank of 16,384 interactions. Frozen controllers operated a four-track simulator on 192 held-out layouts. A pair counted as successful only if the controller achieved two opposite, unseen goals after the same probe. At 128 labels, JSTP achieved 71.69% paired success. Disconnecting its consequence-to-action path while retaining both training losses reduced success to 66.75% (difference: 4.94 percentage points; 99% paired layout-bootstrap interval [3.97, 5.85]). In an expanded comparison, a pretrained ACT adaptation did better at 128 labels (73.78%); at 1,024 labels ACT and JSTP reached 94.58% and 94.90%, respectively. JSTP thus approached ACT on this apparatus with 18,304 parameters trained from scratch, but was not the best sparse-label method. The matched comparison supports a contribution from consequence feedback; task success alone does not establish emergent object segmentation.

**Keywords:** joint state transition prediction; goal-conditioned control; action supervision; functional grouping; latent intervention.

[Preprint DOI (v1.2)](https://doi.org/10.5281/zenodo.24392467) · [Revised English PDF](JSTP-Goal-Generalization-Ke-Yue.pdf) · [Full English text](manuscript-en.md) · [Chinese edition](manuscript.md)

Full theory: [English](../../../theory/JSTP-Theory.en.md) · [Chinese original](../../../theory/JSTP-详细理论稿-2026-11-19.md)
