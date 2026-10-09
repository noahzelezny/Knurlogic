"""interfaces/page/ -- the page a person opens.

  server.py        `knurlogic ui`: the page with nothing loaded, its routes
  documents.py     the documents the page and /status.json are built on,
                   shared with `serve`
  nodes.py         the machines it sees: PEERS, discovery, /status.json
  peers.py         what it does with peers: the gate, residency, settings
  messages.py      the protocol messages it answers (peer_table)
  loads.py         load and unload: here, on a peer, or across machines
  router.py        one address for every model here and on peers
  relay.py         a peer page reaching a model this machine started
  peek.py          another server's settings: /peek and /apply
  prompt_cache.py  /v1/prompt-cache, forwarded to the model's server
  hub.py           Hugging Face search and download for the picker
  updates.py       is a model, or knurlogic, out of date
  assets/          the page itself (index.html), shipped as package data
"""
