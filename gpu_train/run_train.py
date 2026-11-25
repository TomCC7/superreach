#!/usr/bin/env python3
"""Train the 50 Hz tri-mode SuperReach policy with MuJoCo Warp and PyTorch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
import xml.etree.ElementTree as ET

RUNTIME = Path(__file__).resolve().parent / 'runtime'
BANK_SHA = '50c0b4f33b6ea55ce6018601d62d998720c7157773db313caa53bfde12943b62'
SCENE_SHA = 'd74db7c5df991c8a5cea7d8f982fe344c3d3d981526e29a532d26ff8b99fd7be'
PAYLOAD_SHA = 'eee09201a52fbcec4cf9138ba67ca25fc67f588e81abd0627366f26fac002cee'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def find_pinned(root, pattern, digest):
    matches = [p for p in sorted(root.rglob(pattern)) if p.is_file() and sha256(p) == digest]
    if not matches:
        raise ValueError(f'Missing registered {pattern} (SHA-256 {digest}) under {root}. '
                         'Run python scripts/download_artifacts.py first.')
    return matches[0]


def inputs(data):
    archive = data / 'trajectories'
    manifest = find_pinned(archive, 'MANIFEST.json', BANK_SHA)
    scene = find_pinned(data / 'robot', '*.xml', SCENE_SHA)
    payload = find_pinned(data / 'catalog', '*.npz', PAYLOAD_SHA)
    for row in json.loads(manifest.read_text())['rows']:
        shard = (manifest.parent / row['file']).resolve()
        if not shard.is_relative_to(archive.resolve()) or not shard.is_file():
            raise ValueError(f'Missing or invalid trajectory shard: {row["file"]}')
    root = ET.parse(scene)
    compiler = root.find('compiler')
    meshes = [item.attrib['file'] for item in root.findall('./asset/mesh') if 'file' in item.attrib]
    candidates = [scene.parent, *(data / 'robot').rglob('meshes')]
    mesh_dir = next((p for p in candidates if all((p / name).is_file() for name in meshes)), None)
    if mesh_dir is None:
        raise ValueError(f'No complete mesh directory found under {data / "robot"}')
    compiler.set('meshdir', str(mesh_dir.resolve()))
    return root, manifest.parent, payload


def train(args, recipe, torch):
    recipe.SEED = args.seed
    recipe.RECIPE['arch'] = args.arch
    recipe.RECIPE['updates'] = args.updates
    b = recipe.setup(args.arch)
    b.T.seed(args.seed)
    torch.set_num_threads(4)
    l = recipe.make(b, 4096, True, 256, 512)
    from evaluation import evaluate

    out = args.out_dir
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    cadence = l.e.num_envs / l.wm.cfg.num_batches / l.wm.cfg.batch_size / l.wm.cfg.horizon * 16
    b.T.atomic(out / 'activation.json', dict(recipe=recipe.RECIPE, seed=args.seed,
               worlds=4096, wm_batch=256, actor_batch=512, source_digest=b.eef_source_digest))
    t0 = time.monotonic()
    n = 0
    last_eval = 0
    with (out / 'metrics.jsonl').open('x', buffering=1) as metrics:
        while not stopped and n < args.updates:
            stats = l.collect()
            wanted = int(max(0, l.count['real_steps'] - 16) * cadence)
            while not stopped and l.count['wm_calls'] < wanted:
                stats.update(l.wm_update())
                if l.count['wm_calls'] <= 20:
                    continue
                stats.update(recipe.policy_step(b, l))
                l.finite()
                n = l.count['actor_updates']
                status = dict(count=l.count.copy(), metrics=stats, elapsed_seconds=time.monotonic()-t0,
                              last_completed_eval=last_eval, peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
                b.T.atomic(out / 'status.json', status)
                metrics.write(json.dumps(status, allow_nan=False) + '\n')
                if n == 1 or n % 50 == 0:
                    print(json.dumps(status, allow_nan=False), flush=True)
                if n % recipe.RECIPE['eval_every'] == 0 or n == args.updates:
                    recipe.save(b, l, out, f'actor{n}')
                    evaluate(b, l, out, n)
                    last_eval = n
                    b.T.atomic(out / 'status.json', status | {'last_completed_eval': n})
                if n == args.updates:
                    break
    if stopped and n != args.updates:
        recipe.save(b, l, out, 'stopped')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--out-dir', type=Path, required=True, help='New run directory')
    parser.add_argument('--arch', choices=('dual_encoder', 'masked'), default='dual_encoder')
    parser.add_argument('--updates', type=int, default=15000)
    parser.add_argument('--seed', type=int, default=20261118)
    parser.add_argument('--check-data', action='store_true', help='Check registered inputs without CUDA or writing outputs')
    parser.add_argument('--verify-cpu', action='store_true', help='Run the four-world native CPU diagnostic instead of training')
    args = parser.parse_args()
    if args.updates < 1 or not 0 <= args.seed < 2**32 - 2:
        parser.error('--updates must be positive and --seed must be in [0, 2**32-3]')
    try:
        scene, bank, payload = inputs(args.data_dir.resolve())
    except (OSError, ValueError, ET.ParseError) as error:
        parser.error(str(error))
    if args.check_data:
        print(f'Registered inputs found under {args.data_dir.resolve()}. Training verifies each shard digest during loading.')
        return
    try:
        import torch
        from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
    except ImportError as error:
        parser.error(f'GPU training dependencies unavailable: {error}. See requirements-gpu.txt and docs/gpu-training.md.')
    if not args.verify_cpu and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Install CUDA-enabled PyTorch and expose a compatible NVIDIA GPU.')
    if args.verify_cpu and torch.cuda.is_available():
        parser.error('Run the CPU diagnostic with CUDA_VISIBLE_DEVICES=')
    args.out_dir = args.out_dir.resolve()
    if args.out_dir.exists() or args.out_dir.is_symlink():
        parser.error(f'output already exists: {args.out_dir}')
    namespace = RUNTIME / 'openyam-50hz-policyik-eef-v1' / 'tri-mode-dual-enc-dr10ms-joint075-15k-v2'
    with TemporaryDirectory(prefix='superreach-scene-') as tmp:
        model = Path(tmp) / 'runtime.xml'
        scene.write(model, encoding='utf-8')
        os.environ.update(SUPERREACH_SCENE=str(model), SUPERREACH_BANK=str(bank), SUPERREACH_PAYLOAD=str(payload))
        sys.path.insert(0, str(namespace))
        import train as recipe
        args.out_dir.mkdir(parents=True, exist_ok=False)
        if args.verify_cpu:
            from verification import verify
            recipe.SEED = args.seed
            b = recipe.setup(args.arch)
            b.T.seed(args.seed)
            torch.set_num_threads(4)
            learner = recipe.make(b, 4, False, 4, 4, True)
            receipt = verify(b, learner, False, args.out_dir)
            b.T.atomic(args.out_dir / 'verify.json', receipt)
            print(f'CPU diagnostic passed: {args.out_dir / "verify.json"}')
        else:
            train(args, recipe, torch)


if __name__ == '__main__':
    main()
