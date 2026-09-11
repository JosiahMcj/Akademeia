# Akademeia

My curriculum machine, v3 -- the scale ladder at greater scale. Successors:
Paideia (v1) asked whether curricula buy speed; Ephebeia (v2) asked what they
buy at a fixed budget and whether tiny-scale rankings survive a 3.4x scale-up
(they did, exactly). Akademeia asks the questions those two could not reach
at small scale: does the ranking survive an ORDER OF MAGNITUDE, where does the
chained-multiplication depth wall break, does the structural-generalization
lottery ever narrow, and what does a scratchpad buy?

[PROTOCOL.md](PROTOCOL.md) was the first commit, before any code. The
substrate is PyTorch, running on whichever accelerator or processor is
available; every law, gate, and statistic carries from v2 unchanged, and the
bench pins prove the port is byte-faithful. Campaigns wait on my per-device
determinism and golden-parity gates.

```
make setup && make test
```
