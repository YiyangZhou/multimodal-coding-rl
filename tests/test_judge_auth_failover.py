import asyncio
import unittest
import urllib.error
from unittest.mock import patch
from mmrl.reward import QualityJudge, InfrastructureError, JudgeAuthenticationError
from mmrl.audit import infrastructure_failure_code

class AuthFailoverTests(unittest.IsolatedAsyncioTestCase):
    def judge(self):
        with patch.dict('os.environ', {'OPENAI_API_KEYS': 'bad-a,bad-b,good'}):
            return QualityJudge(concurrency=2)

    async def test_same_input_and_bad_keys_stay_disabled(self):
        judge=self.judge();calls=[]
        def grade(key,*args):
            calls.append((key,args))
            if key!='good':raise JudgeAuthenticationError('rejected')
            return 'score'
        judge._grade=grade
        args=('task','candidate',b'png',[b'reference'])
        for _ in range(4):self.assertEqual(await judge.grade(*args),'score')
        self.assertEqual(judge._disabled_keys,{'bad-a','bad-b'})
        self.assertTrue(all(x[1][:4]==args for x in calls))
        self.assertEqual(sum(k=='bad-a' for k,_ in calls),1)
        self.assertEqual(sum(k=='bad-b' for k,_ in calls),1)

    async def test_all_rejected_bounded_and_never_zero_reward(self):
        judge=self.judge();calls=[]
        def grade(key,*args):calls.append(key);raise JudgeAuthenticationError('rejected')
        judge._grade=grade
        with self.assertRaisesRegex(InfrastructureError,'All configured'):
            await judge.grade(None,'',b'',[])
        self.assertEqual(len(calls),3)

    async def test_other_errors_not_quarantined(self):
        judge=self.judge();calls=[]
        def grade(key,*args):calls.append(key);raise InfrastructureError('transport')
        judge._grade=grade
        with self.assertRaisesRegex(InfrastructureError,'transport'):
            await judge.grade(None,'',b'',[])
        self.assertEqual(len(calls),1)
        self.assertFalse(judge._disabled_keys)

    def test_diagnostics_do_not_copy_exception_payload(self):
        self.assertEqual(infrastructure_failure_code('Quality judge HTTP 429; exclude/retry this measurement'),'judge_http_429')
        for value in ['https://host/?token=private','Quality judge HTTP 401; exclude/retry this measurement private',None,{}]:
            self.assertEqual(infrastructure_failure_code(value),'other_infrastructure')
