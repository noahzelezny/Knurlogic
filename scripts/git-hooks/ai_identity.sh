# Sourced by pre-commit and pre-push: what counts as AI attribution.
# AI_IDENT: an author or committer name/email that is an AI agent's.
# AI_NAMED: a commit message naming an AI model as its author or reviewer
# ("Fable 5.1's review", "Sonnet built"); "Claude Code" the client and the
# Anthropic API are knurlogic's subject and stay allowed.
AI_IDENT='anthropic|claude|openai|copilot|codex|gemini|(^|[^a-z])(fable|opus|sonnet|haiku)([^a-z]|$)'
AI_NAMED='^co-authored-by:.*(claude|anthropic)|generated with \[?claude|(fable|opus|sonnet|haiku) ?[0-9]|claude (fable|opus|sonnet|haiku)|(fable|opus|sonnet|haiku) (built|wrote|reviewed)|(fable|opus|sonnet|haiku)'"'"'s (re-?review|review|post-build)|(fable|opus|sonnet|haiku) review|by (fable|opus|sonnet|haiku|claude)([^a-z]|$)'
