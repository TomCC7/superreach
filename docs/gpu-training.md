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

The default run starts cold with 4,096 worlds, 256 world-model samples per minibatch and 512 actor starts per update. It uses ten 2 ms physics substeps per control tick, sensor noise and bias, 0–10 ms command delay, payload randomization up to 2 kg, and a 32-tick differentiable actor rollout. `--arch masked` selects the alternative goal encoder. Control the run budget with `--updates` and RNG initialization with `--seed`. The run directory must be new and receives metrics, checkpoints and native MuJoCo evaluations. See [data generation](../data_gen/README.md) for the separate generator workflow.
