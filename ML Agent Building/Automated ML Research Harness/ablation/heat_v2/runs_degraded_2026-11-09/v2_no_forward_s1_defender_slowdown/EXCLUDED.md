Excluded from the ablation (2026-11-09).
Screen stage ran 10:36:54-10:39:42 while Windows Update + a Defender security-intelligence update ran
(10:34-10:39, System/Defender event logs). Every screen job took 28-93 s vs 2-8 s in the other runs,
so wall-time compute accounting (548/400 s) and the projected-cost pruning were invalid
(torch_mlp, screen 0.031, was pruned on projected cost; ridge 0.079 became the final model; test 0.0635).
Redone as runs/v2_no_forward_s1.
