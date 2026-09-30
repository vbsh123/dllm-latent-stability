# OpenCompass GSM8K scorer provenance

Copyright 2020 OpenCompass Authors. Apache-2.0; see LICENSE.
Source copied unmodified from https://github.com/open-compass/opencompass/blob/e2d2b9a5236ba6a2f0a855af93796e4489c268b7/opencompass/datasets/gsm8k.py
Revision: e2d2b9a5236ba6a2f0a855af93796e4489c268b7. SHA256 hashes in provenance.json. Upstream root NOTICE was absent.

opencompass_score.py selects the original dataset postprocessor, prediction
postprocessor and Gsm8kEvaluator AST nodes. Function/method bodies are unchanged;
registry decorators and unrelated imports/classes are omitted; BaseEvaluator is
substituted with object because the selected methods do not use its behavior.
This allows offline scoring without installing a second inference framework.

This is the standard GSM8K extraction/equality path, not a verified CreditDecoding
configuration. No prompt template, runner-level preprocessing or shot/split settings
are imported. Its comma/fraction/last-number limitations are intentionally preserved.
