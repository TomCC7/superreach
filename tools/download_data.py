#!/usr/bin/env python3
"""Download and unpack the SuperReach training dataset (no model files)."""
import argparse
from pathlib import Path, PurePosixPath
import tarfile
from tempfile import TemporaryDirectory

REPO = 'playercc7/superreach-data'
ARCHIVE = 'trajectory_bank_16k.tar'


def unpack(archive: Path, destination: Path) -> None:
    # Publish only a complete extraction. Never merge with a previous bank.
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'{destination} already exists; use a new data directory')
    with TemporaryDirectory(prefix='.unpack-', dir=destination.parent) as staging:
        root = Path(staging) / 'bank'
        root.mkdir()
        seen = set()
        with tarfile.open(archive, 'r:') as bundle:
            for member in bundle:
                name = PurePosixPath(member.name)
                if (name.is_absolute() or '..' in name.parts or name in seen
                        or not (member.isfile() or member.isdir())):
                    raise ValueError(f'Unsafe or duplicate archive member: {member.name!r}')
                seen.add(name)
                bundle.extract(member, root, filter='data')
        if not any(root.rglob('MANIFEST.json')):
            raise ValueError('Trajectory archive does not contain MANIFEST.json')
        root.rename(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--revision', default='main', help='Dataset branch, tag, or commit SHA')
    args = parser.parse_args()
    target = args.data_dir.resolve()
    bank = target / 'trajectories'
    if bank.exists() or bank.is_symlink():
        parser.error(f'{bank} already exists; use a new data directory')
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        parser.error('install huggingface_hub before downloading')
    try:
        snapshot_download(repo_id=REPO, repo_type='dataset', revision=args.revision,
                          local_dir=target, allow_patterns=[ARCHIVE, 'catalog/**', 'robot/**'])
        for directory in ('catalog', 'robot'):
            if not (target / directory).is_dir():
                raise ValueError(f'Dataset snapshot is missing {directory}/')
        unpack(target / ARCHIVE, bank)
    except HfHubHTTPError as error:
        status = error.response.status_code if error.response is not None else 'unknown'
        parser.exit(1, f'Dataset download failed (HTTP {status}). Check access to {REPO} and run hf auth login if needed.\n')
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f'Dataset setup failed: {error}\n')
    print(f'{REPO} ({args.revision}) downloaded to {target}')
    print(f'Unpacked trajectory bank: {bank}')


if __name__ == '__main__':
    main()
