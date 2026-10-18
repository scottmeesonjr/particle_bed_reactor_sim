# Runaway demonstration

Commit 59f43bd. Ji et al. element at 8 MW/L, uniform power, channel effects off (Kerrebrock's setup). Slice 10 starts 20 K hotter than steady; nothing else changes. Model: docs/model.md section 5, flows readjusting instantly.

## Result
- **With the cold frit:** settled; the disturbance decays (eigenvalue -2.55 1/s, so it shrinks by a factor of e every 0.39 s).
- **Without a cold frit:** runaway. The hot slice loses flow, heats further, and its outlet gas passes the 3,800 K fuel limit at 7.6 s and leaves the 5000 K property range at 9.1 s (eigenvalue +0.47 1/s: the disturbance grows by a factor of e every 2.1 s at first).
