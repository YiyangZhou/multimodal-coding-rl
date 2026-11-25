import asyncio, copy, sys, types
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tarfile
import tempfile
import unittest
from mmrl import opencode_eval as overlay
from mmrl import opencode_exit_diagnostics as diag

class Diagnostics(unittest.TestCase):
    def test_sanitization(self):
        secret = 'private prompt https://private.example/token'
        with tempfile.TemporaryDirectory() as td:
            db=Path(td)/'db';con=sqlite3.connect(db)
            con.executescript('create table session(id text,parent_id text); create table message(session_id text,data text,time_created integer); insert into session values("root",NULL);')
            con.execute('insert into message values(?,?,?)',('root',json.dumps({'role':'assistant','finish':'tool-calls','error':{'name':secret,'data':{'statusCode':400,'message':secret}}}),1));con.commit();con.close()
            out=diag.collect(db,-15,False,('permission denied '+secret).encode())
            self.assertIsNotNone(diag.validate(out));self.assertEqual(out['root_error'],'other');self.assertEqual(out['http_status'],400)
            self.assertTrue(out['stderr_flags']['denied']);self.assertNotIn(secret,json.dumps(out))
            out['prompt']=secret;self.assertIsNone(diag.validate(out))
    def test_failure_is_not_outcome(self):
        with tempfile.TemporaryDirectory() as td:
            db=Path(td)/'missing';value=diag.collect(db,1,True,b'')
            self.assertEqual(value['collection'],'db_missing')
            db.write_text('invalid sqlite');self.assertEqual(diag.collect(db,1,False,b'')['collection'],'db_error')
            diag.save(Path(td)/'nonexistent'/'out',db,1,False,b'')
    def test_reject_unknown_source(self):
        with self.assertRaises(ValueError):overlay.patch_source(b'# unknown')

    def test_harness_case_association(self):
        out=diag.collect('/missing',0,False,b'')
        original={'status':'failed','output':[{'text':'private fixture token'}],'metadata':{'sandbox_reward':'0.4'}}
        calls=[]
        class Harness:
            config=types.SimpleNamespace(agent='opencode')
            def _build_gym_tar(self):raise AssertionError
            def _box_path(self,h,p):return h+p
            async def _download_json(self,h,p):
                calls.append((h,p))
                return copy.deepcopy(original if p.endswith('/response.json') else out)
        module=types.ModuleType('responses_api_agents.harness_agent.app');module.HarnessAgent=Harness
        sys.modules[module.__name__]=module
        overlay.install()
        response=asyncio.run(Harness()._download_json('case-A','case-A/work/response.json'))
        self.assertEqual(response['status'],original['status']);self.assertEqual(response['output'],original['output'])
        self.assertEqual(response['metadata']['sandbox_reward'],'0.4')
        self.assertEqual(json.loads(response['metadata']['opencode_exit_diagnostics']),out)
        self.assertEqual(calls[-1],('case-A','case-A/work/'+diag.FILENAME))
        out['bad']='private fixture token'
        response=asyncio.run(Harness()._download_json('case-B','case-B/work/response.json'))
        self.assertEqual(json.loads(response['metadata']['opencode_exit_diagnostics'])['collection'],'unavailable')
        self.assertNotIn('private fixture token',response['metadata']['opencode_exit_diagnostics'])

@unittest.skipUnless(os.environ.get('MMRL_TEST_GYM_ROOT'),'requires pinned integrated Gym source')
class Archive(unittest.TestCase):
    def test_stdin_and_diagnostics_archive(self):
        original=(Path(os.environ['MMRL_TEST_GYM_ROOT'])/overlay.TARGET).read_bytes()
        self.assertIn(hashlib.sha256(original).hexdigest(),(overlay.ORIGINAL_SHA,overlay.STDIN_SHA))
        for stdin in (False,True):
            patched=overlay.patch_source(original,stdin_prompt=stdin);compile(patched,'candidate','exec')
            aa={n.name:ast.dump(n) for n in ast.walk(ast.parse(original)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
            bb={n.name:ast.dump(n) for n in ast.walk(ast.parse(patched)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
            self.assertEqual(sum(aa[k]!=bb[k] for k in aa),1)
            if stdin:
                self.assertIn(b'proc.communicate(piped_prompt.encode("utf-8"))',patched)
                self.assertNotIn(b'cmd.append(prompt)',patched)
            with tempfile.TemporaryDirectory() as td:
                path=Path(td)/'src.tar.gz'
                with tarfile.open(path,'w:gz') as tf:
                    member=tarfile.TarInfo(overlay.TARGET);member.size=len(original);tf.addfile(member,io.BytesIO(original))
                overlay.rewrite_archive(path,stdin_prompt=stdin)
                with tarfile.open(path) as tf:self.assertEqual(tf.extractfile(overlay.TARGET).read(),patched)

if __name__=='__main__':unittest.main(verbosity=2)
