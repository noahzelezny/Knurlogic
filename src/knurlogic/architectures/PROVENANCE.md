# Vendored architecture modules

Each entry is a claim that this file is the arithmetic the
artifacts were validated against -- not merely that it imports.


## qwen4_exp.py

- taken: 2026-09-18
- from: `/Volumes/Thunderbay SSD/venvs/qwen4exp/lib/python3.12/site-packages/mlx_lm/models/qwen4_exp.py`
- interpreter: `/Volumes/Thunderbay SSD/venvs/qwen4exp/bin/python`
- mlx-lm: 0.32.0
- sha256: `15df2080b3197db26067e1e4e23c2ce152841a8f0f60ff1fb2b2976e93dbb4b9`
- note: the env vqlab fits and scores qwen4_exp artifacts in; carries PipelineMixin and the predicate-arity shim

## qwen3_5.py

- taken: 2026-09-18
- from: `/Volumes/Thunderbay SSD/venvs/qwen4exp/lib/python3.12/site-packages/mlx_lm/models/qwen3_5.py`
- interpreter: `/Volumes/Thunderbay SSD/venvs/qwen4exp/bin/python`
- mlx-lm: 0.32.0
- sha256: `14c4898a03567998e825cb1817942001871e979b9e0cefd3b4383cbbb61eddf3`
- note: qwen4exp venv (mlx-lm 0.32.0) taken as authoritative: it is where vqlab fits and scores, and it is the SUPERSET -- qwen3_5 here carries PipelineMixin, the exo-env copy does not

## qwen3_5_moe.py

- taken: 2026-09-18
- from: `/Volumes/Thunderbay SSD/venvs/qwen4exp/lib/python3.12/site-packages/mlx_lm/models/qwen3_5_moe.py`
- interpreter: `/Volumes/Thunderbay SSD/venvs/qwen4exp/bin/python`
- mlx-lm: 0.32.0
- sha256: `ef9e8e1f6a5c097b29587c8330e8eb9c9cbdc52fbb4597fbc2362606c1996619`
- note: qwen4exp venv (mlx-lm 0.32.0) taken as authoritative: it is where vqlab fits and scores, and it is the SUPERSET -- qwen3_5 here carries PipelineMixin, the exo-env copy does not

## gemma4_text.py

- taken: 2026-09-18
- from: `/Volumes/Thunderbay SSD/venvs/qwen4exp/lib/python3.12/site-packages/mlx_lm/models/gemma4_text.py`
- interpreter: `/Volumes/Thunderbay SSD/venvs/qwen4exp/bin/python`
- mlx-lm: 0.32.0
- sha256: `f3f8c047c2ac31306267e8de61bb06d2952f2bf3fd4adbbb8ce25b952ccac01f`
- note: qwen4exp venv (mlx-lm 0.32.0) taken as authoritative: it is where vqlab fits and scores, and it is the SUPERSET -- qwen3_5 here carries PipelineMixin, the exo-env copy does not
