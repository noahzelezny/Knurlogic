# Finding the other machines without exo (design, 2026-09-24, for review)

Today `cluster.inventory(EXO_URL)` is the only source of "which machines
exist": no exo, no second machine. This replaces that source and leaves
exo, when it is running, as one more witness rather than the authority.
Scope: discovery, identity and reachability only. Running a model across
machines (`knurlogic node`, the ring/jaccl pipeline) is the next step and
builds on this.

## 1. Standard name, standard mechanism

DNS-SD over multicast DNS (Bonjour), service type **`_knurlogic._tcp`**.
It is what every Mac already runs, what AirPlay and printers use, and it
needs no daemon, no dependency and no configuration.

* **Register**: every `knurlogic ui` / `serve` bound to a non-loopback
  address registers `_knurlogic._tcp` on THAT interface only (the
  interface index of the bound address), port = its HTTP port. A process
  bound to loopback registers nothing: advertising what nobody can reach
  is the M4 stumble again.
* **TXT record**: `id` (stable node id, below), `name` (ComputerName),
  `ver` (knurlogic version), `schema` (status SCHEMA), `role` (ui/serve).
* **Browse + resolve**: `DNSServiceBrowse` -> `DNSServiceResolve` ->
  `DNSServiceGetAddrInfo` (IPv4 first, link-local v6 kept as fallback).

Implementation: ctypes against the `dns_sd.h` API in libSystem
(`DNSServiceRegister`, `Browse`, `Resolve`, `GetAddrInfo`,
`DNSServiceRefSockFD`, `DNSServiceProcessResult`), one daemon thread
running `select` over the refs' sockets. No subprocess parsing of the
`dns-sd` CLI (its output format is not an interface). Same approach as the
temperature reader: stdlib only, and a failure is "no peers found", never
a crash.

## 2. Identity

A node is its **id**, not its name: names collide ("MacBook Pro") and get
renamed. `id` = first 12 hex of sha256(IOPlatformUUID) -- stable across
reboots and addresses, and not the raw hardware UUID on the wire. `name`
is display only (`scutil --get ComputerName`, e.g. "Studio A").
exo's node records are matched to ours by name, then by model/chip, and
only for showing exo's placements.

## 3. Sources, merged

`inventory()` becomes: **self + Bonjour + manual + exo**, deduplicated by
id, each node carrying `found_by: ["bonjour", "manual", "exo"]`.

* **Manual**: `--peer HOST[:PORT]` (repeatable) for networks where
  multicast is blocked. A peer that ever answered is remembered in
  `~/.knurlogic/peers.json` (id, name, last address), so the second run
  needs no flags. Remembered peers that stop answering are shown as
  "not answering since <time>", not dropped silently.
* **exo**: consulted only if it answers. A node only exo knows about is
  still drawn from exo's figures, as today.

## 4. Reachability is named, on both ends

The failure tonight was silence. Every peer gets one of:

| state | meaning | said where |
|---|---|---|
| `answering` | status fetched | -- |
| `found, not answering` | advertised, HTTP times out: almost always the macOS firewall on THAT machine | this page AND that machine's page (it can see inbound attempts never arrive? no -- see open question 2) |
| `answering, different version` | schema/ver differ | both |
| `remembered, gone` | in peers.json, not advertised, not answering | this page |

The fix text is concrete: which machine, which setting, which binary
(the interpreter path, since that is what the firewall lists).

## 5. Binding

Default stays **loopback** (a model endpoint should not appear on the
network by accident). New: `--host cluster` binds the Thunderbolt and
wired Ethernet addresses (never Wi-Fi unless named) and advertises on
them. `--host ADDR` / `0.0.0.0` stay as today.

## 6. Where it lives

New package `knurlogic/cluster/`: `discovery.py` (dns_sd via ctypes),
`peers.py` (manual + remembered + merge + reachability), `identity.py`.
`interfaces/cluster.py` keeps the exo front end and CLI and imports these;
the future `knurlogic node` agent lands in the same package. `machine/`
stays "facts about THIS box".

## 7. Tests (no network needed)

* Register and browse on `kDNSServiceInterfaceIndexLocalOnly`: a service
  registered in-process is found, resolved, and its TXT parsed.
* Merge: the same id from bonjour + manual + exo is one node with three
  `found_by`; two nodes with the same name and different ids stay two.
* Reachability states from stubbed fetches (timeout, version mismatch).
* The engine-import tripwire still holds (none of this imports mlx).

## Open questions

1. `--host cluster` vs making advertise-on-Thunderbolt the default for
   `ui` only (not `serve`). the maintainer's call tomorrow.
2. The firewall: can the BLOCKED machine detect it on its own? It sees its
   own listener but not the dropped connections. Candidate: each node
   probes its own non-loopback address from itself (the firewall blocks
   that too -- observed tonight on the M4) and reports "my own address
   does not answer me: firewall". Cheap and local; needs checking that
   self-connections are filtered the same way on every macOS in use.
3. Resending the firewall prompt when missed (the maintainer): re-asking needs the
   app's firewall entry removed, which is admin. Probably: detect (2),
   then name the exact System Settings path; never touch the setting.

## a review review (2026-09-24): build it, with these changes -- accepted

1. **The blocked machine learns it is blocked FROM ITS PEERS.** Its
   outbound works: it fetches a peer's status, which already says "M4:
   found, not answering", and reports "the Studio sees me and cannot
   connect: the firewall on THIS machine". A self-probe is a secondary
   hint only, unmeasured across macOS versions. `socketfilterfw
   --getglobalstate/--listapps` is READ to name the exact binary (the
   framework's `Python.app`, not the venv symlink); never written.
2. **Local Network privacy (Sequoia+) is a second silent blocker.** A
   browse that returns nothing is reported as "no results in N s; check
   Privacy -> Local Network for <terminal app>", not "no peers".
3. **Bind 0.0.0.0 and filter by the accepted connection's local address**
   (`getsockname()`), not one socket per address: the Thunderbolt Bridge
   re-addresses on replug. `cluster` = Thunderbolt Bridge only; wired
   Ethernet must be named, with a printed warning that POSTs are
   unauthenticated. HELD for the maintainer (it changes what is exposed).
4. **One advertisement per machine: `ui` registers, `serve` does not**;
   `serve_port` goes in TXT. Instance name `<ComputerName> <id[:6]>` so two
   "MacBook Pro"s never autorename to "(2)". Answers open question 1.
5. **ctypes rules**: callbacks and refs owned by the discovery object;
   never deallocate a ref inside its callback; honour MoreComing and
   Add/remove; key browse results by (instance, interfaceIndex) and pass
   that index to Resolve/GetAddrInfo. mDNSResponder owns the registration
   and sends goodbyes -- the reason a stdlib mDNS is worse. `dns-sd -B
   _knurlogic._tcp` is printed as the diagnostic, not parsed.
6. **IPv4 only in v1** (urllib cannot carry `%bridge0` zone ids; the
   bridge self-assigns 169.254 anyway).
7. **Identity via IOKit ctypes** (IOPlatformExpertDevice -> IOPlatformUUID),
   cached; lives in `machine/identity.py` -- a fact about this box. The
   hashed id is a persistent LAN identifier, and says so.
8. **Merge**: key `id`; exo matched by IP, then name; `peers.json`
   versioned, keyed by id, atomic write, last address updated; self is
   recognised in its own browse by id.
9. **`cluster/` holds every source** -- `exo.py` moves there from
   `interfaces/cluster.py`, which becomes CLI only.
10. `knurlogic doctor` runs the cluster checks; a gone peer's text says
    "asleep?"; the dns_sd tests are macOS-only.

Build order: identity -> reachability + `--peer` + peers.json + the
cross-check (fixes tonight with no Bonjour) -> register -> browse/resolve
-> `--host cluster` (the maintainer) -> exo demoted to `cluster/exo.py` -> doctor.

## Built and measured (2026-09-25, Studio 192.0.2.1 <-> M4 192.0.2.2 over Thunderbolt)

Steps 1-4 and 6 are built (`machine/identity.py`, `cluster/peers.py`,
`cluster/discovery.py`, `cluster/exo.py`); `--host cluster` (5) waits for
the maintainer, `doctor` (7) is next.

* With no `--peer` and no remembered peers (fresh `KNURLOGIC_HOME`) the
  Studio found the M4 by Bonjour alone, fetched its status, and the M4
  learned the Studio from the introduction header: both `answering`.
* **mDNS over the Thunderbolt link was ONE-WAY.** The Studio sees the
  M4's services on en4; the M4 sees nothing of the Studio's on en3 --
  not knurlogic, not `_ssh`, not `_smb` (`dns-sd -B` on each side). Not
  a knurlogic fault, and exactly why introductions exist: one direction
  of discovery is enough for both machines to know each other. It also
  means a pair where NEITHER direction works needs `--peer` once; after
  that the peer is remembered.
