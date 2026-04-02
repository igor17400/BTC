"""Keras evaluation utilities.

Evaluation logic is now handled directly by the runner via eval_fn/test_fn
closures that build Keras dataloaders and call model.fast_evaluate().
This module is kept for any future evaluation helpers.
"""
