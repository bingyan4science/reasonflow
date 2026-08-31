"""Anchored CoT trajectory modules.

Modules:
    dataset_utils     -- cot_step_positions helper for anchor extraction
    anchor_extraction -- extract_anchors (middle-layer pooling of a finetuned LM)
    suffix_decoder    -- SuffixDecoder that reads out cot[k:] + ans from an anchor
"""
