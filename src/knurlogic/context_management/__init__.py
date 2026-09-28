"""context_management/ -- what the model sees of a long conversation.

  context_edits.py  history surgery over OpenAI-shaped messages: the
                    clearing edits and folding a resent compaction back in,
                    without a model
  compaction.py     server-side compaction: when a request's history is
                    compacted, and the summary pass that does it

Pure history surgery and compaction, model-agnostic: messages in, messages
out, callable by any harness -- the HTTP servers in interfaces/http are one
caller. No mlx, no HTTP. Depends only on tuning/settings (the
KNURLOGIC_COMPACT_* knobs).
"""
