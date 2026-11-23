import csv
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from app import RunStore, make_handler


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.store=RunStore(Path(self.temp.name))
        self.server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(self.store))
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.base=f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join();self.store.close();self.temp.cleanup()

    def request(self,path,body=None,headers=None):
        request=Request(self.base+path,data=json.dumps(body).encode() if body is not None else None,
                        headers=headers or {'Content-Type':'application/json'})
        with urlopen(request,timeout=10) as response:
            return response.status,response.headers,response.read()

    def wait_done(self,run_id):
        for _ in range(200):
            job=self.store.get(run_id)
            if job['status'] in ('completed','failed','cancelled'):
                return job
            time.sleep(.01)
        self.fail('Timed out waiting for test experiment')

    def test_dashboard_static_and_graph_endpoints(self):
        for path in ('/','/styles.css','/app.js','/readme.md','/api/graphs'):
            status,_,data=self.request(path)
            self.assertEqual(status,200);self.assertTrue(data)
        _,_,data=self.request('/api/graphs')
        self.assertEqual(len(json.loads(data)['graphs']),6)
        with self.assertRaises(HTTPError) as error:
            self.request('/../app.py')
        self.assertEqual(error.exception.code,404)

    def test_validation_rejects_bad_potential_and_cross_origin(self):
        with self.assertRaises(HTTPError) as error:
            self.request('/api/runs',{'q':-1})
        self.assertEqual(error.exception.code,400)
        self.assertIn('Negative potentials',error.exception.read().decode())
        with self.assertRaises(HTTPError) as error:
            self.request('/api/runs',{},headers={'Origin':'http://unrelated.example'})
        self.assertEqual(error.exception.code,403)
        _,_,data=self.request('/api/validate',{'graph':'paper9','q':.1})
        self.assertAlmostEqual(json.loads(data)['config']['dt'],1/1.1)

    def test_run_persistence_and_exports(self):
        status,_,data=self.request('/api/runs',{'N':500,'T':5,'q':.1,'name':'API test'})
        self.assertEqual(status,202)
        run_id=json.loads(data)['id']
        self.assertEqual(self.wait_done(run_id)['status'],'completed')
        _,_,data=self.request('/api/runs/'+run_id)
        result=json.loads(data)['result']
        self.assertEqual(result['config']['name'],'API test')
        _,headers,export=self.request('/api/runs/'+run_id+'/json')
        self.assertIn('attachment',headers['Content-Disposition'])
        self.assertEqual(json.loads(export)['counts'],result['counts'])
        _,_,export=self.request('/api/runs/'+run_id+'/csv')
        rows=list(csv.DictReader(io.StringIO(export.decode())))
        self.assertEqual(len(rows),9*(5+2))
        for source in result['observed']:
            for target in result['observed']:
                pair_rows=[row for row in rows if row['source']==str(source) and row['target']==str(target)]
                self.assertEqual(sum(int(row['count']) for row in pair_rows),500)
        restored=RunStore(Path(self.temp.name))
        try:
            self.assertEqual(restored.get(run_id)['status'],'completed')
            self.assertEqual(restored.result(run_id)['counts'],result['counts'])
        finally:
            restored.close()

    def test_cancel_preserves_no_partial_result(self):
        _,_,data=self.request('/api/runs',{'N':1000000,'T':17})
        run_id=json.loads(data)['id']
        self.request('/api/runs/'+run_id+'/cancel',{})
        self.assertEqual(self.wait_done(run_id)['status'],'cancelled')
        self.assertFalse((Path(self.temp.name)/f'{run_id}.json').exists())


if __name__=='__main__':
    unittest.main()
