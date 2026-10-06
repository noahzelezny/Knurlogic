"""DeepSeek-V4 (deepseek_v4): MLA-style shared-KV attention over a
128-token sliding window plus learned compressed pools (Compressor, and a
top-k Indexer on the ratio-4 layers), hash-routed first layers, mHC
hyper-connections. One MTP head (heads/deepseek_v4.py, a sidecar beside
the trunk; the trunk's sanitize still drops `mtp.*`); Vision-Exp's DSpark
block drafter instead (heads/deepseek_v4_dspark.py, its own sidecar).
Images: the DeepSeek-V4-Flash-Vision-Exp artifact (vision_n_layers > 0),
its tower in vision/ (docs/design/deepseek-vision.md).

The chat template is templates/deepseek_v4.jinja here (the conversion
ships a stub; `chat_templates` below says when it replaces one); its
thinking levels are DeepSeek's three modes: Non-think,
Think High, Think Max (the official prefix), the "deepseek_effort" dialect.
Vision-Exp's variant has four (off, low, high, max): "deepseek_vision_effort".
"""

MANIFEST = {
    "name": "deepseek",
    "architectures": {
        # KV precision: every layer's cache is the module's own
        # DeepseekV4Cache (a bf16 RotatingKVCache window + compressor and
        # indexer pools), which engine/kvquant.py does not know how to
        # quantize. Refused until someone measures it.
        "deepseek_v4": {
            # a tensor split (engine/runtime/tensor.py) knows this trunk.
            # `divisible`: config keys a split must divide by the ranks --
            # wo_a is grouped over whole heads: a rank keeps whole groups
            # (its arrays alone would also divide at 16 ranks, 8 groups)
            "tensor": {"divisible": ["o_groups"]},
            "model_types": ["deepseek_v4"],
            "kv_quant": {"refused": (
                "deepseek_v4 caches through its own DeepseekV4Cache (a "
                "128-token bf16 window plus compressed pools); knurlogic's "
                "KV quantization does not apply to it")},
            "head": dict(
                names=["deepseek_v4"],
                head="knurlogic.engine.families.deepseek.heads."
                     "deepseek_v4:MTPHead",
                # the [B, S, hc, D] streams INTO the trunk's hc_head: the
                # official MTPBlock takes all hc streams, not the collapse
                capture="hc_head", draft_cache="DeepseekV4Cache",
                sidecar_name="mtp-head-mxfp4.safetensors",
                # DeepseekV4Cache cannot trim (its pools); caches.py rolls an
                # untrimmable cache back by its whole state
                cache_semantics="copy",
                # every sidecar key is under mtp.0. (the official names)
                layout=("mtp",),
                # Vision-Exp's DSpark (3 stages under mtp.*, block 5):
                # drafted by engine/mtp/block_loop, captured at the outputs
                # of dspark_target_layer_ids (docs/design/deepseek-vision.md)
                block=dict(
                    head="knurlogic.engine.families.deepseek.heads."
                         "deepseek_v4_dspark:DSparkHead",
                    sidecar_name="mtp-head-dspark-mxfp4.safetensors",
                    config="dspark_block_size")),
            # a pipeline split: the trunk core's class. Nothing to
            # re-index: each block froze its compress ratio, hash routing
            # and RoPE from its GLOBAL layer_id at __init__, every layer's
            # cache is the same DeepseekV4Cache (make_cache is one per kept
            # layer), and the stream between stages is the [B, S, hc, D]
            # hyper-connection state, which every stage builds from its own
            # embedding. The block takes the token ids as a third argument
            # (hash routing); the stage wrappers pass it through.
            "pipeline": {"core": "DeepseekV4Model"},
            # A tensor split (tuning/resolve.tensor_refusals): architecture
            # edit 21 (architecture/PROVENANCE.md) rounds every FP8 / FP4
            # linear's input through act_quant in blocks of 128 along it,
            # as DeepSeek's reference does. A split cuts three of those
            # inputs, so a rank's slice must hold whole blocks, or the split
            # model rounds other blocks than the whole one. Each input's
            # width is the product of its config keys (a key absent with no
            # default: not checked). Edit 20's blocks -- the kv, the pooled
            # rows, the indexer -- run whole on every rank.
            "tensor_split": {
                "act_quant_block": 128,
                "inputs": [
                    {"what": "o_groups x o_lora_rank (wo_b's input)",
                     "keys": ["o_groups", "o_lora_rank"]},
                    {"what": "moe_intermediate_size (routed experts' "
                             "down_proj input)",
                     "keys": ["moe_intermediate_size"]},
                    {"what": "moe_intermediate_size x n_shared_experts (the "
                             "shared expert's down_proj input)",
                     "keys": ["moe_intermediate_size", "n_shared_experts"],
                     "defaults": {"n_shared_experts": 1}},
                ]},
        },
    },
    # Chat templates knurlogic supplies in place of an artifact's own
    # (engine/templates reads this; templates/PROVENANCE.md has the port).
    #   file        the template, relative to this folder
    #   base/prefix a variant: the base's text with `prefix` first
    #   model_types the artifacts (config.json model_type) it serves
    #   stubs       sha256 of artifact templates it replaces outright
    #   marker      a string only this dialect's templates contain (a copy
    #               of knurlogic's own, shipped in a release, is replaced)
    #   when_config the variant is chosen when the artifact's config.json
    #               has any of these keys > 0
    #   parser      the tool-call parser, "module:attr" -> (start, end, fn)
    #   part_separator  what a message's list of text parts is joined with,
    #               per role ("default" for the rest), as the maker's
    #               encoder joins them; "" when absent
    "chat_templates": {
        # Flash: encoding_dsv4.py joins a tool result's text parts with
        # "\n\n" (render_message, tool_result blocks); a user message's
        # list content is not something its encoder takes, so mlx-lm's ""
        "deepseek_v4": {
            "file": "templates/deepseek_v4.jinja",
            "model_types": ["deepseek_v4"],
            # mlx-community/DeepSeek-V4-Flash(-8bit) chat_template.jinja,
            # 2026-09
            "stubs": ["718a756ad62609c2539a4a4c0fa4279306773e47df105674c01f"
                      "10fcb0f18c6e"],
            "marker": "｜DSML｜",
            "parser": "knurlogic.engine.families.deepseek.chat_template:"
                      "PARSER",
            "part_separator": {"tool": "\n\n"},
        },
        # DeepSeek-V4-Flash-Vision-Exp: its encoder is Flash's with other
        # reasoning-effort prefixes (its "high" is Flash's "max", its "max"
        # a new one); the template is Flash's with this line first. Chosen
        # by the artifact's config.json: the ViT (vision_n_layers) or the
        # DSpark drafter (dspark_block_size), which Flash has neither of --
        # a text-only conversion keeps both fields. Its encoder joins every
        # message's content blocks with "\n\n" (images included, text-only
        # ones too) and a tool result's text parts the same.
        "deepseek_v4_vision": {
            "base": "deepseek_v4",
            "prefix": "{%- set dsv4_vision = true -%}\n",
            "when_config": ["vision_n_layers", "dspark_block_size"],
            "part_separator": {"default": "\n\n"},
        },
    },
    "thinking": {
        "deepseek_effort": {
            "detect": {"all": ["thinking_mode", "Reasoning Effort",
                               "enable_thinking"]},
            "default": "high",
            "native": [["none", "off", {"thinking_mode": "chat"}],
                       ["high", "high", {"thinking_mode": "thinking"}],
                       ["xhigh", "max", {"thinking_mode": "thinking",
                                         "reasoning_effort": "max"}]],
        },
        # DeepSeek-V4-Flash-Vision-Exp: four levels, its encoder's
        # REASONING_EFFORT_PROMPTS ("low", its default, adds no prefix);
        # its variant's first line sets dsv4_vision, so it is detected first
        "deepseek_vision_effort": {
            "detect": {"all": ["set dsv4_vision = true", "thinking_mode",
                               "Reasoning Effort", "enable_thinking"]},
            "default": "low",
            "native": [["none", "off", {"thinking_mode": "chat"}],
                       ["low", "low", {"thinking_mode": "thinking",
                                       "reasoning_effort": "low"}],
                       ["high", "high", {"thinking_mode": "thinking",
                                         "reasoning_effort": "high"}],
                       ["xhigh", "max", {"thinking_mode": "thinking",
                                         "reasoning_effort": "max"}]],
        },
    },
    "vision": {"build": "knurlogic.engine.families.deepseek.vision:build",
               "architectures": ["deepseek_v4"],
               # Vision-Exp keeps its vision fields flat at the top of
               # config.json: vision_n_layers > 0. A text-only conversion
               # keeps the fields but not the tower, so the tower's own
               # tensors (`tower_in_weights`) must be in the artifact too.
               # Its ViT and aligner live under these prefixes. The trunk's
               # own vision tensors (`trunk_keys`, name suffixes): every
               # gate's bias_vl and the four image rows. A conversion with
               # none of them loads text-only (vision_n_layers built as 0,
               # engine/vision/registry.vision_weights); with some, it is
               # refused.
               "signature": {"config_layers": "vision_n_layers",
                             "tower_in_weights": "vision.",
                             "tower_prefixes": ["vision.", "aligner."],
                             "trunk_keys": [".gate.bias_vl", "image_start",
                                            "image_end", "image_newline",
                                            "image_pad"]}},
}
