# Joint State Transition Prediction: Goal Generalization with Sparse Action Labels

**Ke Yue (岳轲) · Research manuscript · 19 November 2026**

## Abstract

Joint State Transition Prediction (JSTP) treats an agent’s action and changes in its environment as components of the same transition. It allows a network to predict an action directly or to reuse a partial prediction of its consequences. We test whether this second route helps when goals are revealed after an exploratory interaction and labels for goal-appropriate actions are scarce. Eight methods were fitted to nested annotation budgets of 128, 1,024 and 16,384 on a common bank of 16,384 interaction records, with three training seeds. Frozen policies actuated a four-track simulator on 192 held-out layouts. A trial pair counted as successful only if both opposite, previously unseen goals were achieved from the same probe history. With 128 labels, JSTP achieved 71.69% paired success, compared with 67.45% for a one-step world model and 49.79% for an action-only policy. Disconnecting consequence feedback while retaining both training objectives reduced paired success to 66.75% (difference 4.94 percentage points; 99% paired layout-bootstrap interval 3.97–5.85). The disconnected model slightly outperformed JSTP at both larger budgets. Official SmolVLA fine-tuning and a DINOv2/DINO-WM task adaptation reached 23.74% and 50.58%, respectively, under the sparse-label protocol. The former does not use the optical loss; pretraining and compute are not matched. Functional co-motion was more readily decoded from the disconnected model than from JSTP, separating decodability from successful control. These results support reuse of partial consequences in this apparatus, not a general ranking of VLA and world-model methods or a test of claims about language and consciousness.

**Keywords:** joint state transition prediction, goal-conditioned control, action annotation efficiency, functional grouping, latent intervention.

[Chinese manuscript](manuscript.md) · [Complete comparison workbook](tables/JSTP-八种模型同场对照.xlsx)
