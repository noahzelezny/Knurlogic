# The MCP server

## knurlogic/interfaces/mcp.py

Managing local models by hand means guessing: whether memory has
settled, whether a model fits, what the knobs are and why they are set that
way. An agent guesses worse than a person, and faster. Every tool answers
deterministically, reports what it looked at, and refuses rather than
gambles.

Design rules:

- **`ready` is a gate, not a status line.** Loading while another load is
  still moving memory is a common failure. `ready` names every reason it is
  not, and `load` calls it first and refuses rather than trying anyway.
- **`fit` refuses to be optimistic.** It measures available memory as free
  + inactive (the file cache macOS hands over on demand), because both
  other definitions are wrong in opposite directions -- the footprint sum
  and top's "unused" read 75.9 and 1.6 GiB on a machine with 70 available.
  A model that does not fit is a refusal with the arithmetic attached, not
  a warning somebody scrolls past.
- **`settings` returns every knob with its measurement.** A number without
  its provenance is a number an agent will change for no reason.
- **`load` never switches a model inside a running server.** The settings
  that matter are read at import and compiled into kernel source, so they
  can only be chosen before the process starts. Loading spawns a fresh one.
- **Nothing deletes an artifact or writes to one.** Reading a machine's
  state and starting a server on it are reversible; removing weights is
  not.
