Checkpoint: final.pt Epoch: 49 Model Type: simple Architecture: 1024 -> 2048 -> 1024 -> 512 -> 1 Normalization: orthonorm Activation: groupsort

Quick Inference Snapshot (latest run):

Cosine: 0.529784 -> -0.040982 (-0.570766)
Energy: -377.874603 -> -379.691223 (+1.816620)
Steps executed: 200 / 200
2D slice distance to reference: 0.466529 -> 0.007208 (+0.459321)
Off-plane residual (1024D): 0.000000 -> 0.365938
Note: plotted coordinates are 2D projections; primary metrics remain 1024D.
Checkpoint Metrics (saved during training):

Train Loss: N/A
Eval Loss: N/A
Cosine Before: 0.529784
Cosine After: -0.040982
Improvement: -0.570766
Success Rate: 0.00%
Info: unavailable checkpoint metrics were backfilled from latest runtime inference.
Live Inference Metrics (latest run):

Reference source: dataset
Noise scale: 0.0500
Steps requested/executed: 200 / 200
Early stop: False
Cosine before: 0.529784
Cosine after: -0.040982
Cosine improvement: -0.570766
Energy(clean ref): -374.978638
Energy(start noisy): -377.874603
Energy(final): -379.691223
Energy improvement (start-final): +1.816620
Success (energy descent): True
Success (cosine gain): False
Displacement x_T - x_0: 0.590927
2D slice dist(reference): 0.466529 -> 0.007208 (+0.459321)
Off-plane residual: 0.000000 -> 0.365938
SOTA Batch Eval (latest run):

Eval batch / bank: 256 / 4096
Cosine before/after: 0.532171 -> -0.111175
Cosine improvement mean: -0.64334595
Cosine success rate: 0.00%
L2(clean,x) mean: 0.442857 -> 0.362995
L2 improvement mean: +0.07986206
L2 success rate: 99.22%
Energy improvement mean: +2.10355520
Energy success rate: 100.00%
MMD (RBF): 0.337901
C2ST accuracy: 100.00%
C2ST raw accuracy: 100.00%
PRDC precision/recall: 0.0039 / 0.0000
PRDC density/coverage: 0.0004 / 0.0039
kNN cosine top1 improvement: -0.37013760
kNN L2 improvement: +0.2311857

Inference Results:

Model type: simple
Initial noise scale (relative): 0.0500
Langevin method: pid
Steps requested: 200
Learning rate: 0.001000
Trajectory steps executed: 200
Early stop triggered: False
Forced full steps (GUI debug mode): True
Reference source: dataset
Energy range on scanned plane: [-374.1056, -218.7853]
Energy(clean ref): -374.978638
Energy(noisy start): -377.874603
Energy(denoised/final): -379.691223
Delta energy (final - start): -1.816620
Primary improvement (start - final energy): +1.816620
Delta energy (final - clean): -4.712585
Cosine(clean, noisy): 0.529784
Cosine(clean, final): -0.040982
Cosine improvement: -0.570766
Final displacement ||x_T - x_0||: 0.590927
2D slice distance to reference: 0.466529 -> 0.007208 (+0.459321)
Note: this is 2D projection only; 1024D cosine/energy can differ.
SOTA Batch Eval:

Eval batch / bank: 256 / 4096
Cosine before/after: 0.532171 -> -0.111175
Cosine improvement mean: -0.64334595
Cosine success rate: 0.00%
L2(clean,x) mean: 0.442857 -> 0.362995 (+0.07986206)
L2 success rate: 99.22%
Mean denoise step ||x_T-x_0||: 0.573624
Energy improvement mean: +2.10355520
Energy success rate: 100.00%
MMD (RBF): 0.337901
C2ST accuracy: 100.00%
C2ST raw accuracy: 100.00%
PRDC (P/R/D/C): 0.0039 / 0.0000 / 0.0004 / 0.0039
kNN cosine top1 improvement: -0.37013760
kNN L2 improvement: +0.23118576