# Settings resolution

## knurlogic/machine/preferences.py

Beside the strategy (the default launch preset), two kinds of setting are
not a model's to vary and so are not kept per base model:

- **compaction** (`tuning/settings.COMPACT_KNOBS`) -- how a server compacts
  a long conversation. Policy, not a property of any model. Every model
  server on the machine reads it per request, so a change applies to the
  next request of every running model.
- **per-chip rounding** (`KNURLOGIC_CROSS_CHIP`; on means the same rounding
  on every chip, i.e. per-chip rounding off) -- a property of the cluster's
  hardware mix, not of a model and not of any preset. Read at launch;
  unset is off (per-chip rounding on). The page offers on/off only; `auto`
  is still taken from env / `--set`.

Kept in `~/.config/knurlogic/settings.json` (`XDG_CONFIG_HOME` honoured),
beside the allowance and the strategy; the page's Knurlogic tab applies it
to every machine, each through its own page. What is saved here beats the
same name in a server's environment (a per-model launch value saved before
compaction became knurlogic-wide is ignored rather than left to shadow
it); an explicit `--set` of the cross-chip knob still beats it at that
launch.

## knurlogic/tuning/resolve.py

The resolver owns the final value of every knob. It does not write env
files and it does not hope one wins: an env file that sets a knob and is
sourced before another that assigns the same knob unconditionally silently
benchmarks the second value twice. Anything that resolves settings must
hand back one dict and be the last word on it.

Headroom is an input, not something this package detects. Machine
inventory belongs to whatever manages the machines.

**One machine or several.** A cluster resolves the same knobs against a
different budget per node, because the machines differ and because each
node holds only its shard. So `resolve()` takes either a byte count (one
machine, one `Resolution`) or nodes (a `ClusterResolution`, one
`Resolution` each).

What a node holds is a placement question, and placement belongs to
whatever does the sharding. When nobody says, the shard is assumed
proportional to the node's working set, which is an assumption and is
recorded as a note on every resolution that rides on it, not a
measurement.
