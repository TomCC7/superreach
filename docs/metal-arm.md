# MakerMods Metal Arm — Stage 2 evaluation

[Run Metal Arm in the sandbox](https://playercc7-superreach.static.hf.space/?robot=metal-arm) · [Full native evaluation data](evaluation-data/metal-arm-e6-submm-20k-native.json)

The frozen 20k actor was evaluated on CPU with native MuJoCo 3.11.0 at 50 Hz (ten 2 ms substeps per decision). The clean EEF 6D median across case means is 0.83 mm. The lowest-load case is not used as a substitute for the suite median: at 0 kg the clean HOME case means are 0.39 mm (EEF 6D) and 0.25 mm (XYZ-only). Under full hardware domain randomization, EEF 6D mean / median across case means are 9.60 / 2.54 mm. No case triggered the evaluator’s physical-failure flag (>100 rad/s sampled joint speed) across 129 case/mode runs. That flag does not measure tracking precision or certify stability.

Separate stationary HOME supplement: mean TCP error over the last 100 of 300 post-step ticks (4.02–6.00 s).

| Simulated payload | EEF 6D | Raw PD |
|---:|---:|---:|
| 0 kg | 0.293 mm | 5.968 mm |
| 1.5 kg | 0.210 mm | 16.806 mm |
| 3 kg | 0.276 mm | 27.788 mm |

The Metal Arm simulation uses centered TCP payload inertia merged into the hand. At 3 kg it reaches actuator torque caps; the benchmark is **not a hardware load rating** or hardware safety validation. Moving and step targets have different error distributions from stationary HOME holds.
