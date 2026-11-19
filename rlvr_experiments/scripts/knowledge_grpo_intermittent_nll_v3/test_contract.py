import unittest

from contract import advantages, choose_refs, epoch_batches, reference_tokens, refresh_pool, student_reward


class TestAlternation(unittest.TestCase):
    def test_three_students_only(self):
        for k in range(4):
            rewards=[1]*k+[0]*(3-k)
            actual=advantages(rewards)
            for value,r in zip(actual,rewards):
                self.assertAlmostEqual(value,r-k/3)
            self.assertAlmostEqual(sum(actual),0)
        self.assertEqual(advantages([0,0,0]),[0,0,0])
        self.assertEqual(advantages([1,1,1]),[0,0,0])

    def test_truncation_and_failed_backend(self):
        valid=dict(answer_correct=True,answer_extractable=True,execution_valid=True,truncated=False)
        self.assertEqual(student_reward(valid),1)
        self.assertEqual(student_reward(dict(valid,truncated=True)),0)
        self.assertEqual(student_reward(dict(valid,answer_extractable=False)),0)
        with self.assertRaises(ValueError):
            student_reward(dict(valid,execution_valid=False))

    def test_fifo_cancel_and_reenter_and_rotation(self):
        data={c:{"english_references":[dict(rep=i,rollout_id=f"{c}:{i}") for i in (2,0)]} for c in "abc"}
        def group(q,k):
            return dict(id=q,rewards=[1]*k+[0]*(3-k))
        pending=refresh_pool(["a","b"],[group("a",0),group("c",0)],data)
        self.assertEqual(pending,["a","b","c"])
        pending=refresh_pool(pending,[group("a",1)],data)
        self.assertEqual(refresh_pool(pending,[group("a",0)],data),["b","c","a"])
        chosen=choose_refs(["a","b"],data,{"a":1})
        self.assertEqual([r["rollout_id"] for q,r in chosen],["a:2","b:0"])
        self.assertEqual(choose_refs(["a"],data,{"a":2})[0][1]["rollout_id"],"a:0")

    def test_eos_and_fixed_epochs(self):
        ref=dict(rollout_id="q:en:0",response_token_ids=[1,2])
        self.assertEqual(reference_tokens(ref,7),[1,2,7])
        self.assertEqual(reference_tokens(dict(ref,response_token_ids=[1,2,7]),7),[1,2,7])
        with self.assertRaises(ValueError):
            reference_tokens(dict(ref,response_token_ids=[1,7,2]),7)
        ids=list(range(200))
        plan=epoch_batches(ids,dict(order_seed=20261117,batch_size=16),2)
        self.assertEqual(list(map(len,plan)),([16]*12+[8])*2)
        self.assertEqual(sorted(q for batch in plan[13:] for q in batch),ids)
        self.assertNotEqual(plan[:13],plan[13:])


if __name__=="__main__":
    unittest.main()
