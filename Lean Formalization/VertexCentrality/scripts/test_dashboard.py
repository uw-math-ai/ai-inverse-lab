"""Small checks for graph integrity, source freshness, and serving boundaries."""
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
import project
from project import ROOT, fingerprint, nodes, snapshot
from serve import asset_path

class DashboardTests(unittest.TestCase):
    def test_nodes(self):
        items=nodes()
        self.assertTrue(items)
        self.assertEqual(len(items),len({n['id'] for n in items}))
        self.assertTrue(all(n['body'].strip() and n['source'] for n in items))
    def test_routes(self):
        self.assertEqual(asset_path('/'),ROOT/'dashboard/index.html')
        for path in ('/.git/config','/api/../.env','/blueprint/../../.git/config',
                     '/blueprint/%2e%2e/%2e%2e/.env','/blueprint/%2fetc/passwd'):
            self.assertIsNone(asset_path(path))
    def test_source_changes_invalidate_fingerprint(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in ('lean-toolchain','lakefile.toml','lake-manifest.json','Main.lean'):
                (root/name).write_text('initial')
            with patch.object(project,'ROOT',root):
                before=project.fingerprint()
                (root/'Main.lean').write_text('changed')
                self.assertNotEqual(before,project.fingerprint())
                after=project.fingerprint()
                (root/'README.md').write_text('documentation only')
                self.assertEqual(after,project.fingerprint())
    def test_evidence(self):
        self.assertEqual(len(fingerprint()),64)
        self.assertEqual(snapshot()['branch'],'owen-working-branch')

if __name__=='__main__':unittest.main()
