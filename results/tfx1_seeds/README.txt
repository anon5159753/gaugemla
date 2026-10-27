true_fisher_x1_seeds (2026-10-15): the 1% true-Fisher GaugeMLA init refitted with 4 more label seeds
(20262014 / 20263014 / 20264014 / 20265014; the rollout's is 20261014), t = 0 only, 3 budgets x 7 models.
<kit>/r<seed>/G<G>_<pct>pct.jsonl  the t = 0 row (KL to the teacher, KL to the original, CE, batteries)
<kit>/r<seed>/manifest.json        that seed's stamp data manifest (same eval docs + train stream as the rollout)
<kit>/r<seed>/reference.json       that stamp's reference row
<kit>/init/r<seed>_cells.json      the init's accounting for that seed
<kit>/init/r<seed>/B<B>_*.json     each checkpoint's sidecars (B = latent width)
t0_rows.tsv                        every t = 0 KL: the 4 seeds here + the rollout's row 0 (source = rollout)
names.tsv                          short path -> original path (laptop grid/<kit>/results/...) -> sha256
The Fisher fits (.npz) are not included (the kits' pull_results.sh never takes them); they stay on cluster under
grid/<kit>/fisher/r<seed>/.

t = 0 KL to the teacher, 5 label seeds:
slm1p6b_urx1      25%  n 5  mean 0.536  sd 0.002  min 0.534  max 0.538
slm1p6b_urx1    12.5%  n 5  mean 0.917  sd 0.008  min 0.907  max 0.927
slm1p6b_urx1    6.25%  n 5  mean 2.126  sd 0.007  min 2.117  max 2.135
p410m_urx1        25%  n 5  mean 0.623  sd 0.003  min 0.621  max 0.626
p410m_urx1      12.5%  n 5  mean 1.357  sd 0.010  min 1.345  max 1.371
p410m_urx1      6.25%  n 5  mean 3.423  sd 0.026  min 3.385  max 3.456
q0p5b_tx1         25%  n 5  mean 0.882  sd 0.004  min 0.877  max 0.885
q0p5b_tx1       12.5%  n 5  mean 1.682  sd 0.019  min 1.659  max 1.702
q0p5b_tx1       6.25%  n 5  mean 3.999  sd 0.048  min 3.923  max 4.037
olmo1b_tx1        25%  n 5  mean 2.282  sd 0.083  min 2.142  max 2.348
olmo1b_tx1      12.5%  n 5  mean 2.429  sd 0.078  min 2.304  max 2.491
olmo1b_tx1      6.25%  n 5  mean 3.133  sd 0.049  min 3.091  max 3.214
l3_1b_tx1         25%  n 5  mean 3.857  sd 0.086  min 3.734  max 3.974
l3_1b_tx1       12.5%  n 5  mean 4.879  sd 0.058  min 4.822  max 4.966
l3_1b_tx1       6.25%  n 5  mean 5.646  sd 0.053  min 5.601  max 5.711
tl1b_tx1          25%  n 5  mean 3.320  sd 0.018  min 3.301  max 3.347
tl1b_tx1        12.5%  n 5  mean 3.701  sd 0.015  min 3.682  max 3.718
tl1b_tx1        6.25%  n 5  mean 4.223  sd 0.020  min 4.198  max 4.239
sm360m_tx1        25%  n 5  mean 2.479  sd 0.030  min 2.456  max 2.514
sm360m_tx1      12.5%  n 5  mean 3.261  sd 0.022  min 3.235  max 3.287
sm360m_tx1      6.25%  n 5  mean 4.314  sd 0.023  min 4.279  max 4.342
