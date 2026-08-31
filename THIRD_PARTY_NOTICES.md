# Third-Party Notices

## Meta DiT

[`src/reasonflow/models/dit_canonical.py`](src/reasonflow/models/dit_canonical.py)
is adapted from the `models.py` implementation in Meta Platforms' DiT,
commit [`f4e74c1ec80e8`](https://github.com/facebookresearch/DiT/blob/f4e74c1ec80e8/models.py).
The adapted file preserves portions of the original structure and helper code;
the file-level docstring describes the ReasonFlow modifications.

Copyright (c) Meta Platforms, Inc. and affiliates. The adapted file is
licensed under [Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0)](https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt)
and is not covered by the repository's MIT license. The noncommercial
restriction therefore applies to that file.

## Datasets

Dataset files are not included in this repository. The preprocessing recipe
references:

- [GSM8K](https://github.com/openai/grade-school-math), whose repository
  includes its own license.
- [GSM8K-Aug formulas-only data](https://github.com/da03/implicit_chain_of_thought/tree/main/data/gsm8k),
  which users must obtain from the upstream source and use under its terms.

These datasets are not covered by the ReasonFlow code license.
