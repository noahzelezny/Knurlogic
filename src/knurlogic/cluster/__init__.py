"""The other machines: finding them, knowing them, and saying why one is
silent.

Every SOURCE of "which machines exist" lives here, and none of them is the
authority on its own (docs/DISCOVERY.md):

  peers.py   peers named with --peer, peers remembered from an earlier
             run, and peers that introduced themselves by asking for our
             status -- each with a named reachability state.

exo, Bonjour and the `knurlogic node` agent join this package as they are
built; `interfaces/` keeps only the command lines and pages that use them.
Nothing here imports mlx.
"""
