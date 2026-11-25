# Data generation

For the registered 16,000-trajectory bank, use the [SuperReach dataset](https://huggingface.co/datasets/playercc7/superreach-data) and the [downloader](../tools/download_data.py). The training launcher checks the original artifact hashes. Generating a new bank does not reproduce those registered bytes.

Regeneration uses native RoboPlan Python bindings with RRT, path shortcutting and TOPPRA, plus NumPy, MuJoCo and Pillow. The seed generator records preview videos using EGL, `ffmpeg` and `ffprobe`. Set `SUPERREACH_GEN_DIR` to a separate workspace containing the pinned OpenYAM description under `openyam-description/openyam_single/`. See [`NOTICE`](../NOTICE) for the revision.

```sh
export SUPERREACH_GEN_DIR="$PWD/data/generated"
python data_gen/prepare_model.py
python data_gen/generate_bank.py
python data_gen/bulk16k.py
```

`prepare_model.py` builds the runtime and reduced planning scenes. `generate_bank.py` creates the seed bank. `bulk16k.py` validates native trajectories and expands the seed bank to the full train/dev split. For parallel expansion, use `bulk16k.py --init-only` to initialize the expansion ledgers, then `pool_generator.py` instead of running serial expansion to completion. Do not run both writers at once.

To regenerate the continuous payload catalog, supply the runtime scene and original canonical five-row catalog. The generator keeps those rows and adds deterministic inertia variants. Floating-point results may vary across MuJoCo and linear-algebra builds.

```sh
python data_gen/generate_payload_catalog.py --scene "$SCENE" \
  --canonical "$CANONICAL" --output ./data/generated/continuous_variants.npz
```
