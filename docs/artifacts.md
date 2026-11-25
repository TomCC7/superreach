# Local artifacts

The Git checkout contains source code. If you have access to the linked Space, you can open the interactive demo without a local setup. For a local build, set `ARTIFACTS` to a directory containing the exact inputs listed in `web/assets/BUNDLE.json` and `web/assets/vendor.json`:

- `assets/scene.xml` and `assets/meshes/` contain the 2 ms OpenYAM MuJoCo plant and its meshes. Set `SCENE="$ARTIFACTS/assets/scene.xml"` for the Python commands. The plant loads meshes relative to the scene.
- `models/js/` contains the pinned descriptors, JavaScript graphs, and binary weights. The build verifies descriptors and their graph/weight digests. Do not copy private checkpoints into the site directory.
- `vendor/` contains the pinned MuJoCo JS/WASM and Three.js files with their license texts.

`tools/build_web.py` copies only listed assets to a new directory and makes the scene's mesh directory relative after checking its original hash. It never writes to the artifact directory. Supply inputs you have permission to use; the source license does not cover robot assets or checkpoints.

The Python export command expects a tensor-only PyTorch actor state dictionary or `{'actor': state_dict}`. It checks FK buffers against the supplied scene before loading weights. It emits a single inline-weight ONNX17 graph, `[1,318]` to `[1,6]`. The runtime input is 16 × 19 history values, 7 pose values (quaternion WXYZ), 6 joint-target values, and 1 mode code. The result is six joint-position commands in radians.

`python tools/export_onnx_to_jsbin.py LOCAL_SITE` compiles compatible ONNX files in a scratch site tree with `assets/BUNDLE.json` and `models/`. It validates each ONNX digest and ABI, writes content-addressed descriptors, graphs, and float32 weights, and updates the scratch manifest. The static builder is for the matching pinned demo artifacts. A custom policy needs its own manifest and selector entry.

For the browser test, install `playwright` and its Chromium browser in the verification environment. Serve the assembled site and run:

```sh
python tests/browser_smoke.py --onnx "$ARTIFACTS/models/policy_trimode_dual_enc_10ms_13500.onnx"
```

To refresh the README preview, pass `--screenshot docs/preview.png` and review the image before committing it. Keep runtime assets out of the source release.
