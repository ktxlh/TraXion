"""Ablation pretraining objectives (A7 causal, A8 STM masked modeling).

Drop-in alternative pre-training to the default denoising objective. The
backbone (FeatureEncoder + HumorStack) is identical to the full Model so
that finetune scripts can load the saved checkpoint via strict=False.
"""
