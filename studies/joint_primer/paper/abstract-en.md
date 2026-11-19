# Joint State Transition Prediction: Goal Generalization with Sparse Action Labels

Ke Yue · Independent Researcher · 19 November 2026

## Abstract

A controller can learn what an action does without being told when to take it. Does that knowledge help once a new goal is specified? Joint State Transition Prediction (JSTP) treats an agent's action and changes in its environment as components of the same transition. We test a network that can predict its action directly or use a latent prediction of a local consequence. Eight methods were trained at three action-label budgets on a fixed bank of 16,384 interactions. Frozen policies operated a four-track simulator on 192 held-out layouts. Each trial pair required the policy to achieve two opposite, previously unseen goals from the same probe history. With 128 labels, JSTP achieved 71.69% paired success, versus 67.45% for a one-step world model and 49.79% for an action-only policy. Disconnecting the consequence-to-action path while retaining both training losses reduced paired success to 66.75% (difference: 4.94 percentage points; 99% paired layout-bootstrap interval: 3.97 to 5.85). The disconnected network scored slightly higher at both larger budgets and had more readily decoded co-motion at 128 labels. The results isolate a benefit of using predicted consequences under sparse action supervision in this apparatus. They do not establish that better control requires more readily decoded object representations.

**Keywords:** joint state transition prediction; goal-conditioned control; action supervision; functional grouping; latent intervention.

[Published preprint](https://doi.org/10.5281/zenodo.24366436) · [English PDF](JSTP-Goal-Generalization-Ke-Yue.pdf) · [Full English text](manuscript-en.md) · [Chinese edition](manuscript.md)
