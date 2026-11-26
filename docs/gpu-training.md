# GPU training

Use Linux x86-64, Python 3.13 and an NVIDIA CUDA 13.0-capable setup. The reference environment is listed in [`requirements-gpu.txt`](../requirements-gpu.txt). Start with a separate venv from the CPU demo:

```sh
python3.13 -m venv .venv-gpu
. .venv-gpu/bin/activate
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-gpu.txt
```

Install a compatible [mjlab](https://github.com/mujocolab/mjlab) source build that exposes `mjlab.sim.Simulation`, `SimulationCfg` and `MujocoCfg` for MuJoCo Warp 3.11. The reference run used a source checkout, not a released mjlab wheel. The released mjlab 1.6.1 package pins `rsl-rl-lib==5.4.2`, so it does not resolve with the historical pins in this file. If using a source build, install its dependencies with metadata compatible with the pinned stack and check `python -m pip check`. A clean install has not been verified.

## Data and training

```sh
python scripts/download_artifacts.py
python gpu_train/run_train.py --data-dir ./data --out-dir ./runs/tri_mode
```

The downloader fetches only the [SuperReach dataset](https://huggingface.co/datasets/playercc7/superreach-data). It keeps the tar file and extracts it into `data/trajectories`, alongside `data/robot` and `data/catalog`. The archive must preserve the manifest's relative trajectory paths, including the sibling seed bank. Plan for disk space for both the archive and its extracted contents. Use `--revision` to select a dataset commit. If the dataset requires access, authenticate with `hf auth login` before retrying.

For data admission without allocating GPU memory, add `--check-data` to the training command. The launcher verifies the registered scene, payload catalog and trajectory manifest by SHA-256 and checks that all referenced shards exist. The learner verifies each shard's digest when it loads the bank. The XML's mesh directory is relocated only in a temporary copy. Use `CUDA_VISIBLE_DEVICES` to choose the GPU, and keep at least 12 GiB free for checkpoint writes.

The portable CLI starts a **fresh baseline** with 4,096 worlds, 256 world-model samples per minibatch and 512 actor starts per update. It does not restore or recreate the published E6 checkpoint. It uses ten 2 ms physics substeps per control tick, sensor noise and bias, 0–10 ms command delay, payload randomization up to 2 kg, and a 32-tick differentiable actor rollout. `--arch masked` selects the alternative goal encoder. Control the run budget with `--updates` and RNG initialization with `--seed`. The run directory must be new and receives metrics, checkpoints and native MuJoCo evaluations. See [data generation](../data_gen/README.md) for the separate generator workflow.

## Published E6 recipe

The demo's 20k policy continued the 15k E6 actor for 5,000 updates. The [recipe source](../gpu_train/runtime/openyam-50hz-policyik-eef-v1/e6_submm/) includes the actor, randomized plant parameters and learner hooks, with the effective run settings in `recipe.json`:

- A 16-frame history encoder predicts a 16D latent. A training-only auxiliary head regresses 71 normalized domain parameters (payload, hardware and delay) with MSE weight 0.01. The deployed actor does not take privileged parameters as inputs.
- Keep the existing 8-frame world-model pass and add a 33-frame, 32-transition open-loop pass using recorded effective commands. Of each 64-window long batch, 48 windows are unconditioned and 16 are selected for low peak joint speed. Reject reset-crossing windows rather than using terminal padding.
- Use a smooth position penalty and an ungated `exp(-‖Δp‖² / (2 mm)²)` bonus, weighted 1.5 in EEF and 2.0 in XYZ. Regularize joint speed, joint acceleration and issued-command slew. Actor LR decays from `5e-5` to `5e-6` on the continuation clock; critic and world-model LRs stay at `2e-4`.

These source hooks have component tests on CPU. The historical full-state checkpoint and its restore launcher are not distributed or exercised here. See [evaluation](evaluation.md) for native results.
