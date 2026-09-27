"""GLM-5.3-Flash (glm5_next): vendored from mlx-vlm 0.6.17, the version the
released rungs were built on. Vision tower and an MTP head.
"""

MANIFEST = {
    "name": "glm5",
    "architectures": {
        "glm5_next": {
            "host": "mlx_lm",
            "model_types": ["glm5_next_text", "glm5_next"],
            # MLA: the cache is the compressed latent + rope key and the DSA
            # indexer's keys, in mlx-vlm's own cache classes. The vendored
            # mlx-vlm carries kv_quant/turboquant for its own generate loop;
            # knurlogic's batch engine has never run them, and a latent is
            # not the per-head K/V engine/kvquant.py stores.
            "kv_quant": {"refused": "GLM caches an MLA latent and the DSA "
                                    "indexer's keys in mlx-vlm's own cache "
                                    "classes, not per-head K/V; its vendored "
                                    "kv_quant/turboquant path is not wired "
                                    "into the batch engine"},
            "prefill_chunk": (2048, "34 deltanet layers hold per-token "
                                    "recurrent intermediates (16.8 MB/layer) "
                                    "across a chunk; 4096 OOMed both boxes "
                                    "of a 224 GB pair on the 3.6bpw "
                                    "(2026-09-01, before the per-chunk eval "
                                    "fix); 2048 is the post-fix value"),
            # The head is upstream `layers.45`: a plain-residual DeepSeek-
            # style block (NoPE MLA + DSA indexer + 288-expert MoE + its own
            # shared_head.norm) -- see heads/glm5.py for why it is NOT the
            # trunk's hc DecoderLayer. capture="norm": the final-norm INPUT
            # is the mean-collapsed (B, S, D) hidden that hnorm/eh_proj
            # consume. draft_cache is vestigial: the head class provides
            # make_draft_cache() (CacheList(main-KV, indexer-KV)).
            # cache_semantics="reassign": check_snapshot_semantics True on
            # the loaded 2.7bpw trunk (M4, 2026-09-02).
            # Measured in vqlab on one box, not on a cluster: acceptance
            # 0.8516 pooled (12 prompts x 128 tokens, q6 head, 2.7bpw, M4,
            # 2026-09-02); 1.05x end to end.
            "head": dict(
                # the VLM wrapper's config says glm5_next, the bound
                # LanguageModel's TextConfig says glm5_next_text
                names=["glm5_next", "glm5_next_text"],
                head="knurlogic.engine.families.glm5.heads.glm5:MTPHeadGlm5",
                capture="norm", draft_cache="KVCache",
                sidecar_name="mtp-head-q6.safetensors",
                cache_semantics="reassign",
                layout=("eh_proj", "enorm", "hnorm", "final_norm")),
        },
    },
    # GLM-5.3's template: reasoning_effort in {low, high}, anything else is
    # "max" (the default). It has no off KWARG; "off" is the template's own
    # format for a turn without thinking -- the think block already closed
    # (engine/serve/thinking.CLOSE), at low effort. Measured on 2.7 before
    # it was offered: 12/12 right, 0 reasoning tokens.
    # Its three levels are the ladder's low/medium/high under other names
    # (the maintainer, 2026-09-26): "high" is its middle, "max" its top -- and the
    # default, which is why a harness that asks nothing reasons for 64k.
    # xhigh has no level of its own and goes to max, the highest there is.
    "thinking": {
        "glm_effort": {
            "detect": {"all": ["reasoning_effort", "Reasoning Effort"]},
            "default": "max",
            "native": [["none", "off", {"reasoning_effort": "low",
                                        "_knurlogic_close_think": True}],
                       ["low", "low", {"reasoning_effort": "low"}],
                       ["medium", "high", {"reasoning_effort": "high"}],
                       ["high", "max", {"reasoning_effort": "max"}]],
        },
    },
    "vision": {"build": "knurlogic.engine.families.glm5.vision:build",
               "architectures": ["glm5_next"]},
}
