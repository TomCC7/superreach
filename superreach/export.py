"""Export a tensor-only tri-mode checkpoint to ONNX and check CPU parity."""
import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from .actor import HOME
from .plant import load_actor, load_model


class Wrapper(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, obs):
        return self.actor.rollout(obs[:, :304].reshape(-1, 16, 19), obs[:, 304:311], obs[:, 311:317], obs[:, 317].long())


def export(scene, checkpoint, output):
    model = load_model(scene)
    wrapper = Wrapper(load_actor(model, checkpoint)).eval()
    rng = np.random.default_rng(13500)
    x = rng.normal(0, .35, (384, 318)).astype(np.float32)
    h = x[:, :304].reshape(-1, 16, 19)
    h[:, :, :6] += HOME
    h[:, :, 12:18] += h[:, :, :6]
    h[:, :, 18] = rng.integers(0, 2, h.shape[:2])
    h[:, -1, 18] = 1
    x[:, 304:307] += [.3, 0, .35]
    x[:, 307:311] /= np.linalg.norm(x[:, 307:311], axis=1, keepdims=True)
    x[:, 311:317] += HOME
    x[:, 317] = np.repeat(np.arange(3), 128)
    xt = torch.from_numpy(x)
    with torch.inference_mode():
        expected = torch.cat([wrapper(row[None]) for row in xt]).numpy()
        history = torch.cat((xt[:, :304].reshape(-1,16,19), torch.linspace(0,.30,16).expand(len(x),16)[...,None]), -1)
        native = torch.cat([wrapper.actor({'history': history[i:i+1], 'goal_pose': xt[i:i+1,304:311],
                            'goal_q': xt[i:i+1,311:317], 'mode': xt[i:i+1,317].long()}) for i in range(len(x))]).numpy()
        np.testing.assert_array_equal(expected, native)
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        model_path = Path(tmp)/'policy.onnx'
        torch.onnx.export(wrapper, xt[:1], str(model_path), input_names=['obs'], output_names=['q_des'],
                          opset_version=17, dynamo=False, external_data=False)
        raw = model_path.read_bytes()
        onnx.checker.check_model(raw)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        session = ort.InferenceSession(raw, opts, providers=['CPUExecutionProvider'])
        def run(rows):
            return np.concatenate([session.run(['q_des'], {'obs': row[None]})[0] for row in rows])
        actual = run(x)
        delta = float(np.max(np.abs(expected - actual)))
        if not delta < 1e-5:
            raise RuntimeError(f'ONNX parity failed: {delta} rad')
        hidden = x.copy()
        hidden[:128, 311:317] = np.nan
        hidden[128:256, 307:317] = np.nan
        # Joint mode must receive the pose derived from its visible joint reference.
        hidden[256:, 304:311] = np.nan
        np.testing.assert_array_equal(actual, run(hidden))
        with output.open('xb') as f:
            f.write(raw)
    print(f'384 samples; max PyTorch/ONNX difference {delta:.3g} rad; inactive goals unchanged')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    export(args.scene, args.checkpoint, args.output)


if __name__ == '__main__':
    main()
