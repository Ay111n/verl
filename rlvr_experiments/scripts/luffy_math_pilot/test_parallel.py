"""CPU checks for mixed-domain prompts and parallel rollout identity."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import common
class TestParallel(unittest.TestCase):
    def test_knowledge_choices_preserved(self):
        row={"domain":"knowledge","question_bo":"QUESTION","options_bo":{"A":"CHOICE1","B":"CHOICE2"}}
        text=common.prompt_text(row)
        self.assertIn("A. CHOICE1",text);self.assertIn("B. CHOICE2",text)
    def test_math_prompt_preserved(self):
        row={"domain":"math","question_bo":"MATH"}
        self.assertEqual(common.prompt_text(row),common.make_prompt(common.Candidate("math",row,b"")))
    def request(self):
        return {"request_id":"test","adapter_sha256":"hash","prompts":[{"id":str(i//3),"rollout_index":i%3,"seed":i+17,"prompt_token_ids":[i//3+1]} for i in range(12)]}
    def test_exact_reassembly(self):
        def fake_rpc(path,req,timeout):
            return {"adapter_sha256":"hash","elapsed_seconds":1,"records":list(reversed(req["prompts"]))}
        with tempfile.TemporaryDirectory() as d,patch.object(common,"rpc",side_effect=fake_rpc):
            req=self.request();got=common.rpc_parallel(Path(d),req,[Path(d)/str(i) for i in range(3)])
            self.assertEqual(got["records"],req["prompts"])
            self.assertEqual([r["responses"] for r in got["parallel_workers"]],[4,4,4])
    def test_wrong_adapter_rejected(self):
        def fake_rpc(path,req,timeout):return {"adapter_sha256":"wrong","elapsed_seconds":1,"records":req["prompts"]}
        with tempfile.TemporaryDirectory() as d,patch.object(common,"rpc",side_effect=fake_rpc),self.assertRaises(AssertionError):
            common.rpc_parallel(Path(d),self.request(),[Path(d)/"worker"])
    def test_changed_seed_rejected(self):
        def fake_rpc(path,req,timeout):
            rs=[dict(r,seed=0) for r in req["prompts"]]
            return {"adapter_sha256":"hash","elapsed_seconds":1,"records":rs}
        with tempfile.TemporaryDirectory() as d,patch.object(common,"rpc",side_effect=fake_rpc),self.assertRaises(AssertionError):
            common.rpc_parallel(Path(d),self.request(),[Path(d)/"worker"])
if __name__=="__main__":unittest.main()
