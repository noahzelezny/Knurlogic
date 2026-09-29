# Finding the other machines

How a knurlogic page learns which machines exist, who each one is, and
whether it can be reached. Running a model across machines builds on this
(see [server.md](server.md), "Cluster").

Code: `machine/identity.py` (who this machine is), `cluster/discovery.py`
(Bonjour), `cluster/peers.py` (named, remembered and introduced peers,
merge, reachability), `cluster/links.py` (which link, `--host cluster`),
`cluster/checks.py` (`knurlogic doctor --cluster`). `machine/` holds facts
about this box; `cluster/` holds everything about other boxes.

## Bonjour: the standard mechanism

DNS-SD over multicast DNS, service type **`_knurlogic._tcp`**. Every Mac
already runs it (AirPlay and printers use it); it needs no daemon, no
dependency and no configuration.

* **One advertisement per machine**: the page (`knurlogic ui`) registers;
  `serve` does not. It registers on the interface it is bound to, and a
  process bound to loopback registers nothing -- advertising what nobody
  can reach is worse than silence. The instance name is
  `<ComputerName> <id[:6]>`, so two "MacBook Pro"s never autorename to
  "(2)".
* **TXT record**: `id` (node id, below), `name` (ComputerName), `ver`
  (knurlogic version), `schema` (status schema), `role`.
* **Browse -> resolve -> addrinfo**, IPv4 only (urllib cannot carry a
  `%bridge0` zone id, and the Thunderbolt bridge self-assigns 169.254
  anyway). Browse results are keyed by (instance, interface index) and that
  index is passed to resolve and addrinfo: a Thunderbolt peer also seen on
  Wi-Fi is two results, and resolving without the index can return the
  Wi-Fi address for the Thunderbolt one.

Implementation: ctypes against `dns_sd.h` in libSystem (`DNSServiceRegister`,
`Browse`, `Resolve`, `GetAddrInfo`, `DNSServiceRefSockFD`,
`DNSServiceProcessResult`), one daemon thread running `select` over the refs'
sockets. mDNSResponder owns the registration and sends goodbyes, which is why
a stdlib mDNS stack would be worse (a second responder on 5353, stale records
after a crash). The `dns-sd` CLI is never parsed; `dns-sd -B _knurlogic._tcp`
is printed as a diagnostic. ctypes rules that are load-bearing: callbacks and
refs are owned by the discovery object; a ref is never deallocated inside its
own callback; MoreComing and Add/remove are honoured. Any failure is "nothing
found", never a crash, and `status()` says which.

## Identity

A node is its **id**, not its name: names collide and get renamed. `id` =
first 12 hex of sha256(IOPlatformUUID), read through IOKit and cached --
stable across reboots and addresses, and not the raw hardware UUID on the
wire. It is still a persistent LAN identifier, and the code says so. `name`
(`ComputerName`) is display only. A page recognises itself in its own browse
by id.

## Sources, merged

Peers come from four sources, deduplicated by id, each carrying `found_by`:

* **bonjour** -- the browse above.
* **named** -- `--peer HOST[:PORT]` (repeatable), for networks where
  multicast is blocked.
* **remembered** -- a peer that ever answered is kept in
  `~/.knurlogic/peers.json` (versioned, keyed by id, written atomically,
  addresses only), so the second run needs no flags.
* **introduced** -- every status request carries
  `X-Knurlogic-Peer: <id> <port>`, so the receiving page learns the
  requester. One direction of discovery is enough for both machines to know
  each other; this matters because mDNS over a Thunderbolt link can be
  one-way (one side sees the other's services, not the reverse). Introduced
  peers are capped at 32 and forgotten after an hour of silence, since any
  client can send the header.

Two nodes with the same name and different ids stay two nodes.

## Reachability is named

Silence is the failure mode, so every peer has a state:

| state | meaning |
|---|---|
| `answering` | status fetched |
| `not_answering` | known but the request fails, with a `problem`; a remembered peer shows how long it has been gone ("asleep?") and is never dropped silently |
| `version_mismatch` | schema or version differ |

**A blocked machine learns it is blocked from its peers.** Its outbound
connections work: it fetches a peer's status, which says "found, not
answering" about it, and reports that the firewall on THIS machine refuses
the connection. `socketfilterfw --getglobalstate/--listapps` is read (never
written) to name the exact binary the firewall lists -- the framework's
`Python.app`, not a venv symlink.

**Local Network privacy (macOS Sequoia and later)** is a second silent
blocker: a browse that returns nothing is reported as "no results in N s;
check Privacy -> Local Network for <app>". The grant is per binary path, so
a rebuilt environment needs it again.

`knurlogic doctor --cluster` runs these checks, plus sleep on AC power
(`pmset -c sleep 0`): a machine that sleeps drops off the network and every
peer sees it leave and return.

## Binding

The default is **loopback**: a model endpoint must not appear on the network
by accident. `--host cluster` binds every address but answers only on
loopback and the Thunderbolt links, checked per connection against the local
address it arrived on (`getsockname()`), so a replugged bridge that
re-addresses keeps working; a request from elsewhere gets 403 naming the
Thunderbolt address. It advertises only on Thunderbolt. Links are classified
by `networksetup` hardware port ("Thunderbolt" / bridge), not by interface
type (macOS reports the Thunderbolt link as ethernet), and a peer reachable
two ways is kept on Thunderbolt. `--host ADDR` and `0.0.0.0` bind as named;
POSTs are unauthenticated, so exposing wired Ethernet or Wi-Fi is an explicit
choice.

## Tests (no network needed)

* Register and browse on `kDNSServiceInterfaceIndexLocalOnly`: a service
  registered in-process is found, resolved, and its TXT parsed (macOS only).
* Merge: the same id from several sources is one node with several
  `found_by`; same name, different ids stay two.
* Reachability states from stubbed fetches (timeout, version mismatch).
* None of this imports mlx.

## Module notes

### knurlogic/cluster/discovery.py

DNS-SD service type `_knurlogic._tcp`, through the `dns_sd.h` API in
libSystem. mDNSResponder -- the daemon every Mac already runs -- owns the
registration and the multicast; the module only asks it. A stdlib mDNS
stack would be worse: a second responder on port 5353, its own probing and
TTLs, and stale records for an hour after a crash. With mDNSResponder the
registration dies with the process.

- **register** -- one advertisement per machine, from the page (`ui`), on
  the interface it is bound to; `serve` does not advertise. TXT: id, name,
  ver, schema, role.
- **browse -> resolve -> addrinfo** (IPv4), each step on the interface the
  service was seen on: a Thunderbolt peer seen on Wi-Fi too is two browse
  results, and resolving without the index can hand back the Wi-Fi address
  for the Thunderbolt one.

Load-bearing ctypes rules: every CFUNCTYPE and DNSServiceRef is owned by
the Discovery object for as long as the daemon may call it back; a ref is
never deallocated from inside its own callback (it is queued and freed by
the loop); MoreComing and Add/remove are honoured.

A failure anywhere is "nothing found", never a crash, and `status()` says
which: no library, a register error, or a browse that has seen nothing --
which on Sequoia and later is often the Local Network privacy setting of
the app that started knurlogic, named in the message.

### knurlogic/cluster/peers.py

A second Mac can sit silent because a firewall prompt is waiting on its
own screen, with nothing on either machine saying so. So every peer carries
a state, and a state that is not `answering` carries the fix, on the
machine that can apply it:

| state | meaning |
|---|---|
| `answering` | its status came back |
| `not_answering` | known (named, remembered or introduced) but the request failed; `problem` says what the failure looked like |
| `version_mismatch` | answering, with a status schema this one does not read |

A peer that stops answering is kept and shown with how long it has been
gone -- machines sleep -- never dropped silently.

**How a peer learns about this machine.** Every status request carries an
introduction header (`X-Knurlogic-Peer: <id> <port>`); the receiving page
records the requester's address with that port as an `introduced` peer. So
naming a machine on one side is enough for both to know each other, and the
side that cannot be reached still finds out: it asks its peers what they
see, and a peer that lists it as `not_answering` is a measured fact --
"they can see me and cannot connect", which on a Mac is almost always the
application firewall on this machine.

`peers.json` (`~/.knurlogic/peers.json`) is keyed by node id, versioned,
and written atomically. It holds addresses, not secrets.
