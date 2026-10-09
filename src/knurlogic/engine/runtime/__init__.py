"""knurlogic's own server runtime (docs/design/server.md): the pieces that touch
mlx -- the model host, the scheduler and its memory guard, the executor,
prompts and requests. Serving one model across ranks is engine/split/."""
