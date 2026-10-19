from pathlib import Path
import hashlib
import io
import json
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from gw.assets import file_hash, inventory
from gw.model_packages import load_index, prepare


class ModelPackages(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='gw-package-test-')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root,self.cache = self.base/'project',self.base/'parts'
        (self.root/'artifacts').mkdir(parents=True)
        self.cache.mkdir()
        self.bodies = {'artifacts/checkpoints/test/001.pt':b'first model',
                       'weights/other/001.pt':b'another model'}
        self.items = [dict(destination=name,bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
                      for name,body in self.bodies.items()]
        (self.root/'artifacts/manifest.json').write_text(json.dumps({'checkpoints':
            dict(zip(['test/001','other/001'],self.items))}))
        self.path = self.cache/'models-01.tar'
        self.archive(list(self.bodies.items()))

    def archive(self,entries):
        with tarfile.open(self.path,'w') as output:
            for name,body in entries:
                info = tarfile.TarInfo(name)
                if body is None:
                    info.type,info.linkname = tarfile.SYMTYPE,'../outside'
                    output.addfile(info)
                else:
                    info.size = len(body)
                    output.addfile(info,io.BytesIO(body))
        self.index = dict(status='complete',parts=[dict(file=self.path.name,
                         bytes=self.path.stat().st_size,sha256=file_hash(self.path),models=self.items)])

    def test_prepare_one_experiment_and_reuse_without_archive(self):
        rows = prepare(self.root,self.index,experiment='test',packages_dir=self.cache)
        self.assertEqual(rows[0]['status'],'prepared_from_package')
        self.assertEqual(inventory(self.root,'test')[0]['status'],'verified')
        self.assertFalse((self.root/self.items[1]['destination']).exists())
        self.path.unlink()
        self.assertEqual(prepare(self.root,self.index,experiment='test')[0]['status'],'reused')

    def test_all_registered_models(self):
        self.assertEqual(len(prepare(self.root,self.index,packages_dir=self.cache)),2)
        self.assertTrue(all(row['status']=='verified' for row in inventory(self.root)))

    def test_existing_invalid_model_is_preserved(self):
        target = self.root/self.items[0]['destination']
        target.parent.mkdir(parents=True)
        target.write_bytes(b'user content')
        with self.assertRaises(ValueError):
            prepare(self.root,self.index,packages_dir=self.cache)
        self.assertEqual(target.read_bytes(),b'user content')

    def test_index_identity_duplicate_and_path_changes(self):
        for key,value in [('destination','../outside.pt'),('sha256','0'*64),('bytes',5)]:
            index = json.loads(json.dumps(self.index))
            index['parts'][0]['models'][0][key] = value
            with self.assertRaises(ValueError):
                load_index(self.root,index)
        index = json.loads(json.dumps(self.index))
        index['parts'][0]['models'].append(index['parts'][0]['models'][0])
        with self.assertRaises(ValueError):
            load_index(self.root,index)

    def test_incomplete_index_and_filename_escape(self):
        for status in ['running','failed']:
            with self.assertRaises(ValueError):
                load_index(self.root,dict(self.index,status=status))
        self.index['parts'][0]['file'] = '../escape.tar'
        with self.assertRaises(ValueError):
            load_index(self.root,self.index)

    def test_bad_archive_is_preserved_and_no_model_is_written(self):
        self.path.write_bytes(b'user archive')
        with self.assertRaises(ValueError):
            prepare(self.root,self.index,packages_dir=self.cache)
        self.assertEqual(self.path.read_bytes(),b'user archive')
        self.assertTrue(all(row['status']=='missing' for row in inventory(self.root)))

    def test_unregistered_duplicate_missing_or_linked_members(self):
        normal = list(self.bodies.items())
        for entries in [normal+[('../escape.pt',b'bad')],normal+[normal[0]],
                        normal[:1],[(normal[0][0],None),normal[1]]]:
            self.archive(entries)
            with self.assertRaises(ValueError):
                prepare(self.root,self.index,packages_dir=self.cache)
            self.assertTrue(all(row['status']=='missing' for row in inventory(self.root)))

    def test_changed_tensor_bytes_despite_rehashed_archive(self):
        name,body = list(self.bodies.items())[0]
        self.archive([(name,b'x'*len(body)),list(self.bodies.items())[1]])
        with self.assertRaises(ValueError):
            prepare(self.root,self.index,packages_dir=self.cache)
        self.assertFalse((self.root/name).exists())
        self.assertEqual(list((self.root/name).parent.glob('.gw-model-*')),[])

    def test_absent_file_has_no_fake_download_success(self):
        self.path.unlink()
        with self.assertRaisesRegex(ValueError,'No model package download URL'):
            prepare(self.root,self.index,packages_dir=self.cache)
        with self.assertRaisesRegex(ValueError,'HTTPS'):
            prepare(self.root,self.index,packages_dir=self.cache,urls={self.path.name:'http://example.org/part'})

    def test_https_download_and_invalid_download_cleanup(self):
        body = self.path.read_bytes()
        self.path.unlink()
        class Response(io.BytesIO):
            def geturl(self): return 'https://example.org/models/part'
        with patch('gw.model_packages.urllib.request.urlopen',return_value=Response(b'wrong')):
            with self.assertRaises(ValueError):
                prepare(self.root,self.index,packages_dir=self.cache,urls={self.path.name:'https://example.org/part'})
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.cache.glob('.gw-part-*')),[])
        with patch('gw.model_packages.urllib.request.urlopen',return_value=Response(body)):
            rows = prepare(self.root,self.index,packages_dir=self.cache,urls={self.path.name:'https://example.org/part'})
        self.assertEqual(len(rows),2)

    def test_destination_link_outside_is_rejected(self):
        target = self.root/'artifacts/checkpoints'
        target.symlink_to(self.cache,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'outside'):
            prepare(self.root,self.index,packages_dir=self.cache)

    def test_disk_reserve(self):
        with patch('gw.model_packages.shutil.disk_usage',return_value=SimpleNamespace(free=7*1024**3)):
            with self.assertRaises(RuntimeError):
                prepare(self.root,self.index,packages_dir=self.cache)


if __name__=='__main__':
    unittest.main()
