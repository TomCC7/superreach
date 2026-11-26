"""Assemble a local static site from hash-matched external browser artifacts."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from shutil import copyfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def build(artifacts, output):
    vendor = json.loads((ROOT/'web/assets/vendor.json').read_text())

    def checked(name, digest):
        path = artifacts/name
        if not path.is_file():
            raise ValueError(f'Missing required browser asset: {name}')
        data = path.read_bytes()
        if sha256(data).hexdigest() != digest:
            raise ValueError(f'Hash mismatch: {name}')
        return data

    if output.exists():
        raise ValueError('Output already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        site = Path(tmp)/'site'
        source_files = (
            'index.html', 'style.css', 'robots/openyam/BUNDLE.json',
            'robots/metal-arm/BUNDLE.json', 'robots/metal-arm/LICENSE', 'js/robots.js',
            'js/app.js', 'js/canvas_drag.js', 'js/error_plot.js',
            'js/interaction_config.js', 'js/interactive_physics.js', 'js/interactive_view.js',
            'js/mlp_infer.mjs', 'js/policy.js', 'js/runtime_timing.mjs',
            'js/sim_worker.js', 'js/trajectories.js',
        )
        for name in source_files:
            dest = site/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            copyfile(ROOT/'web'/name, dest)

        def stage(name, digest):
            data = checked(name, digest)
            dest = site/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return data

        for name, digest in vendor.items():
            stage(name, digest)
        for robot in ('openyam', 'metal-arm'):
            base = 'robots/' + robot
            bundle = json.loads((site/base/'BUNDLE.json').read_text())
            for name, digest in bundle['files'].items():
                stage(base+'/'+name, digest)
            for checkpoint in bundle['checkpoints'].values():
                descriptor = checkpoint['js']
                info = json.loads(stage(descriptor['path'], descriptor['sha256']))
                expected_width = {'tri-mode-v2': 318, None: 311}.get(checkpoint.get('abi'))
                if (expected_width is None or info['input'] != expected_width or info['output'] != 6
                        or info['onnx_sha256'] != checkpoint['sha256']):
                    raise ValueError('Compiled policy does not match the pinned model or ABI')
                for field, pattern, key in (('graph', r'graph-[0-9a-f]{20}\.mjs', 'graph_sha256'),
                                            ('weights', r'weights-[0-9a-f]{20}\.bin', 'sha256')):
                    if not re.fullmatch('models/js/'+pattern, info[field]):
                        raise ValueError('Invalid compiled policy asset name')
                    stage(info[field], info[key])
        site.rename(output)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifacts', type=Path, required=True, help='Directory with robots/, models/js/ and vendor/')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    try:
        build(args.artifacts, args.output)
    except ValueError as exc:
        ap.error(str(exc))


if __name__ == '__main__':
    main()
