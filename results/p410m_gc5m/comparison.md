# p410m_gc5m: Pythia-410m 5M, clipped vs unclipped (KL to the teacher)

| construction | KV | t=0 clipped | t=0 unclipped | 5M clipped | 5M unclipped | clipped - unclipped (paired) | change | docs lower |
|---|---|---|---|---|---|---|---|---|
| GQA (pool) | 25% | 4.9414 | 4.9413 | 1.5531 ± 0.0421 | 1.7175 | -0.1643 ± 0.0086 | -9.6% | 64/64 |
| GQA + optimal gauge | 25% | 4.9963 | 4.9961 | 1.2097 ± 0.0339 | 1.4609 | -0.2512 ± 0.0084 | -17.2% | 64/64 |
| Procrustes alignment | 25% | 4.6631 | 4.7009 | 0.8423 ± 0.0280 | 0.9635 | -0.1213 ± 0.0078 | -12.6% | 64/64 |
| Per-group subspace | 25% | 2.1196 | 2.1192 | 0.6592 ± 0.0239 | 0.6798 | -0.0206 ± 0.0025 | -3.0% | 59/64 |
| Joint key, joint value subspace | 25% | 0.9315 | 0.9315 | 0.4781 ± 0.0179 | 0.4712 | +0.0069 ± 0.0010 | +1.5% | 8/64 |
| Decoupled, KL objective, d_h^R = 64 | 25% | 0.4836 | 0.4849 | 0.4130 ± 0.0153 | 0.4049 | +0.0080 ± 0.0013 | +2.0% | 3/64 |
| GaugeMLA, d_h^R = 64 | 25% | 0.6190 | 0.6190 | 0.4265 ± 0.0164 | 0.4197 | +0.0068 ± 0.0020 | +1.6% | 3/64 |
| GaugeMLA, KL width (w=98) | 25% | 0.5624 | 0.5624 | 0.4272 ± 0.0165 | 0.4215 | +0.0058 ± 0.0022 | +1.4% | 8/64 |
| | | | | | | | | |
| GQA (pool) | 12.5% | 4.7475 | 4.7473 | 2.1158 ± 0.0666 | 2.1961 | -0.0803 ± 0.0025 | -3.7% | 64/64 |
| GQA + optimal gauge | 12.5% | 4.7930 | 4.7933 | 1.5797 ± 0.0460 | 1.8336 | -0.2539 ± 0.0108 | -13.8% | 64/64 |
| Procrustes alignment | 12.5% | 5.0428 | 5.0327 | 1.0638 ± 0.0324 | 1.1891 | -0.1253 ± 0.0043 | -10.5% | 64/64 |
| Per-group subspace | 12.5% | 2.8416 | 2.8416 | 0.7705 ± 0.0256 | 0.8061 | -0.0357 ± 0.0018 | -4.4% | 64/64 |
| Joint key, joint value subspace | 12.5% | 1.9495 | 1.9497 | 0.6348 ± 0.0235 | 0.6498 | -0.0150 ± 0.0011 | -2.3% | 62/64 |
| Decoupled, KL objective, d_h^R = 64 | 12.5% | 1.2192 | 1.2195 | 0.5466 ± 0.0213 | 0.5463 | +0.0002 ± 0.0009 | +0.0% | 26/64 |
| GaugeMLA, d_h^R = 64 | 12.5% | 1.3715 | 1.3715 | 0.5659 ± 0.0215 | 0.5653 | +0.0006 ± 0.0027 | +0.1% | 36/64 |
| GaugeMLA, KL width (w=50) | 12.5% | 1.3787 | 1.3776 | 0.5529 ± 0.0216 | 0.5564 | -0.0035 ± 0.0011 | -0.6% | 44/64 |
| | | | | | | | | |
| GQA (pool) | 6.25% | 4.8818 | 4.8817 | 2.4786 ± 0.0715 | 2.5533 | -0.0747 ± 0.0028 | -2.9% | 64/64 |
| GQA + optimal gauge | 6.25% | 4.7769 | 4.7769 | 1.8124 ± 0.0559 | 1.9738 | -0.1614 ± 0.0078 | -8.2% | 64/64 |
| Procrustes alignment | 6.25% | 5.1901 | 5.1903 | 1.2219 ± 0.0347 | 1.3700 | -0.1481 ± 0.0051 | -10.8% | 64/64 |
| Per-group subspace | 6.25% | 3.2411 | 3.2410 | 0.8657 ± 0.0286 | 0.9226 | -0.0569 ± 0.0025 | -6.2% | 64/64 |
| Joint key, joint value subspace | 6.25% | = per-group subspace (one KV head) | | | | | | |
| Decoupled, KL objective, d_h^R = 64 | 6.25% | 3.3943 | 3.3948 | 0.9190 ± 0.0307 | 0.9691 | -0.0501 ± 0.0027 | -5.2% | 64/64 |
| GaugeMLA, d_h^R = 64 | 6.25% | 3.4241 | 3.4241 | 0.9656 ± 0.0319 | 1.0220 | -0.0564 ± 0.0024 | -5.5% | 64/64 |
| GaugeMLA, KL width (w=30) | 6.25% | 2.7835 | 2.7852 | 0.7982 ± 0.0278 | 0.8358 | -0.0376 ± 0.0022 | -4.5% | 64/64 |
| | | | | | | | | |
| MHA control (uncompressed) | 100% | 0.3376 | 0.3375 | 0.3492 ± 0.0137 | 0.3445 | +0.0047 ± 0.0011 | +1.4% | 12/64 |

Lowest 5M KL per budget:

- 25%: clipped Decoupled, KL objective, d_h^R = 64 0.4130; unclipped Decoupled, KL objective, d_h^R = 64 0.4049
- 12.5%: clipped Decoupled, KL objective, d_h^R = 64 0.5466; unclipped Decoupled, KL objective, d_h^R = 64 0.5463
- 6.25%: clipped GaugeMLA, KL width 0.7982; unclipped GaugeMLA, KL width 0.8358

clipped cells: 24; missing: none
clipped runs: grad_clip 1.0 on the token-mean KD loss, card NVIDIA RTX PRO 6000 Blackwell Server Edition; unclipped: the 5M grid (old cluster L40S) and p410m_urx1_5m (RTX PRO 6000) for the three F-reading arms
