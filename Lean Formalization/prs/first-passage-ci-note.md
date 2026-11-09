The current `No Stacked PRs / enforce` failure appears to be infrastructure-wide:
[run 42677115064](https://github.com/TauCetiProject/TauCeti/actions/runs/42677115064)
reports `classification: unknown` with
`pagination exceeds 2000 records: pulls?state=open&sort=created&direction=asc`.

This PR is based directly on main `1e668f5e55bc20f2eb87f4ba49b48833470290d1`,
with one added commit `c325d5de1d` and one new file; it has no unmerged prerequisite.
I have not changed or bypassed the policy machinery. The sandboxed content build
is still pending. Local module build, source guards, header check, all 14 scoped
environment linters, and the local compiled-declaration audit pass.
