import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v6_publication',ROOT/'resources/publish_v6.py')
pub=importlib.util.module_from_spec(spec);spec.loader.exec_module(pub)

class PublicationTests(unittest.TestCase):
    def fixture(self,root):
        (root/'data').mkdir();(root/'private').mkdir();(root/'src').mkdir();(root/'cache').mkdir()
        pub.write(root/'config.json',{'model_path':'/private/qwen','esm_model_path':'/private/esm','scope':'paired public data'})
        pub.write(root/'data/manifest.json',{'source_url':'https://example.org/paired.csv'})
        (root/'private/do_not_publish.key').write_text('private-test-marker')
        (root/'src/model.py').write_text('VALUE=1\n')
        (root/'cache/features.npz').write_bytes(b'private frozen features')

    def test_only_allowlisted_files_and_explicit_preregistration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'source';root.mkdir();self.fixture(root)
            target=Path(directory)/'public'
            files,status=pub.stage(root,target,'preregistered')
            paths={f['path'] for f in files}
            self.assertNotIn('private/do_not_publish.key',paths)
            self.assertNotIn('cache/features.npz',paths)
            self.assertEqual(status['status'],'protocol_registered_not_started')
            self.assertEqual(status['accepted_cases'],0)
            self.assertEqual(pub.read(target/'config.json')['model_path'],'${QWEN3_4B_MODEL_PATH}')
            self.assertNotIn('private-test-marker',''.join(p.read_text() for p in target.rglob('*') if p.is_file()))

    def test_final_scientific_status_comes_from_analyzer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'source';root.mkdir();self.fixture(root)
            pub.write(root/'analysis/summary.json',{'status':'complete_direction_paused','accepted_cases':9,'decision':'pause'})
            _,status=pub.stage(root,Path(directory)/'public','final')
            self.assertEqual(status['status'],'complete_direction_paused')
            self.assertEqual(status['accepted_cases'],9)
            self.assertFalse(status['local_callback_required'])
            self.assertFalse(status['scheduled_monitor'])

    def test_missing_final_analysis_is_technical_not_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'source';root.mkdir();self.fixture(root)
            _,status=pub.stage(root,Path(directory)/'public','final')
            self.assertEqual(status['status'],'technical_inconclusive')
            self.assertFalse(status['analysis_completed'])

    def test_sanitization_covers_dictionary_keys_and_nested_values(self):
        value={'/private/model/path':{'a':['/private/model/file',7]}}
        self.assertEqual(pub.sanitize(value,[('/private/model','${MODEL}')]),{'${MODEL}/path':{'a':['${MODEL}/file',7]}})

    def test_symlinks_cannot_expand_publication_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'source';root.mkdir();self.fixture(root)
            external=Path(directory)/'private_external';external.write_text('external')
            try:(root/'src/leak.py').symlink_to(external)
            except OSError:self.skipTest('Symlink privilege unavailable')
            with self.assertRaisesRegex(ValueError,'symlink'):
                pub.stage(root,Path(directory)/'public','preregistered')

if __name__=='__main__':unittest.main()
