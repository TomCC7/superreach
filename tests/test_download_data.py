import io
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest

from tools.download_data import unpack


class ExtractionTests(unittest.TestCase):
    def test_preserves_bank_layout(self):
        with TemporaryDirectory() as work:
            root = Path(work)
            archive = root / 'trajectories.tar'
            with tarfile.open(archive, 'w') as bundle:
                for name, data in [('bank-full-v2/MANIFEST.json', b'{}'), ('bank/seed.npz', b'fixture')]:
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    bundle.addfile(member, io.BytesIO(data))
            target = root / 'trajectories'
            unpack(archive, target)
            self.assertEqual((target / 'bank' / 'seed.npz').read_bytes(), b'fixture')
            self.assertTrue((target / 'bank-full-v2' / 'MANIFEST.json').is_file())
            with self.assertRaises(FileExistsError):
                unpack(archive, target)

    def test_rejects_traversal_and_links_without_publishing(self):
        for kind, name in [(tarfile.REGTYPE, '../escape'), (tarfile.SYMTYPE, 'link'), (tarfile.LNKTYPE, 'link')]:
            with self.subTest(kind=kind), TemporaryDirectory() as work:
                root = Path(work)
                archive = root / 'trajectories.tar'
                with tarfile.open(archive, 'w') as bundle:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    member.linkname = '../escape'
                    bundle.addfile(member)
                target = root / 'trajectories'
                with self.assertRaisesRegex(ValueError, 'Unsafe or duplicate'):
                    unpack(archive, target)
                self.assertFalse(target.exists())
                self.assertFalse((root / 'escape').exists())


if __name__ == '__main__':
    unittest.main()
