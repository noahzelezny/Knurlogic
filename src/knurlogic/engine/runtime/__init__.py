"""knurlogic's own server runtime (docs/design/server.md): the pieces that touch
mlx. Serving one model across ranks is engine/split/.

  model_host.py      ModelHost: the one model this process serves, and its state
  scheduler.py       the one thread that owns the MLX stream
  memory_guard.py    the scheduler's guard against outgrowing the working set
  executor.py        the seam between the scheduler and what runs a step
  prompt.py          a request's messages -> the prompt's tokens and segments
  request.py         Request: a request's tokens -> the text the client reads
  control_tokens.py  the control-token state machine (reasoning, tool, stop)
  timing.py          what one request took, and where the time went"""
