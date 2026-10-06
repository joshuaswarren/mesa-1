# D3D12 integration i2

Captured Fable snapshot: cc55230bb1e6a05c648d7c43d680b12892e3b729.
This snapshot already contains agx3 82852e1e346 and i1 87c78f894a19.
Integration imports only the focused accepted shader, resource, command, caps,
P09 and P10 deltas, avoiding duplicated rebased Fable/i1 history.

Conflicts preserve both barrier responsibilities: flush deferred compute writes
at API barriers and end only open render passes, retaining compute segment
capacity for long placed-resource loops. Native upload certification retains
cached-stage ownership and common NIR/part/BO cleanup. Native GS stage metadata
is prepared before Poly compilation and its NIR cleanup.

Multiple-fence waits register completion tickets synchronously, retain private
fence lifetime, and use Fable notification parking for ANY. Ticket checks before
and after notification registration cover completions followed by CPU rewinds.
ALL uses ticket waits; these retain the pending GPU timeline wait path while
checking the latched ticket under the fence lock. Shared ANY retains bounded
external progress polling. Native preraster recording stays certificate-only;
this integration does not open physical GS admission.

Qualification commands, complete conflict ledger, immutable package identities,
per-registration raw outputs and machine/corpus stats are recorded in the
private task report and evidence directory d3d12-i2-t1169. Frozen input bytes and
upstream A/B assertions must remain unchanged. Counts describe measured results,
including failures/skips/crashes; compilation/recording is separate from physical
execution evidence.
