"""GLM-5.3-Flash (glm5_next): vendored from mlx-vlm 0.6.17, the version the
released rungs were built on. Vision tower and an MTP head.
"""

MANIFEST = {
    "name": "glm5",
    "architectures": {
        "glm5_next": {
            "host": "mlx_vlm",
            "model_types": ["glm5_next_text", "glm5_next"],
            "prefill_chunk": (2048, "34 deltanet layers hold per-token "
                                    "recurrent intermediates (16.8 MB/layer) "
                                    "across a chunk; 4096 OOMed both boxes "
                                    "of a 224 GB pair on the 3.6bpw "
                                    "(2026-09-01, before the per-chunk eval "
                                    "fix); 2048 is the post-fix value"),
            "head": dict(
                # the VLM wrapper's config says glm5_next, the bound
                # LanguageModel's TextConfig says glm5_next_text
                names=["glm5_next", "glm5_next_text"],
                head="knurlogic.engine.mtp.heads.glm5:MTPHeadGlm5",
                capture="norm", draft_cache="KVCache",
                sidecar_name="mtp-head-q6.safetensors",
                cache_semantics="reassign"),
        },
    },
    "vision": {"build": "knurlogic.engine.vision.glm5:build",
               "architectures": ["glm5_next"]},
}
