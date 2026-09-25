"""The other machines: finding them, knowing them, and saying why one is
silent.

Every SOURCE of "which machines exist" lives here, and none of them is the
authority on its own (docs/DISCOVERY.md):

  peers.py      peers named with --peer, peers remembered from an earlier
                run, and peers that introduced themselves by asking for our
                status -- each with a named reachability state.
  discovery.py  Bonjour: `_knurlogic._tcp`, advertised by a page others can
                reach, browsed by every page.
  exo.py        what exo's /state says about each node: a witness, used
                only for nodes nothing else answers for.

The `knurlogic node` agent joins this package when it is built; `interfaces/` keeps only the command lines and pages that use them.
Nothing here imports mlx.
"""
