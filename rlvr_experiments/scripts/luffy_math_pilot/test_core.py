"""CPU checks for the actual 3+1 objective and stopped sampling probabilities."""
import unittest,math
import torch
from common import advantages,online_loss,demo_loss
class TestObjective(unittest.TestCase):
    def test_all_wrong(self):
        a,d=advantages([0,0,0]);torch.testing.assert_close(a,torch.tensor([-.25]*3));self.assertEqual(float(d),.75)
    def test_all_correct(self):
        a,d=advantages([1,1,1]);self.assertEqual(float(a.abs().sum()),0);self.assertEqual(float(d),0)
    def test_mixed(self):
        a,d=advantages([0,1,0]);torch.testing.assert_close(a,torch.tensor([-.5,.5,-.5]));self.assertEqual(float(d),.5)
    def test_detached_old(self):
        x=torch.tensor([-.5,-2.],dtype=torch.float64,requires_grad=True)
        y=online_loss(x,x,-.5);y.backward();torch.testing.assert_close(x.grad,torch.tensor([.25,.25],dtype=torch.float64))
    def test_clipping(self):
        for ratio,adv in [(1.4,1),(.6,-1)]:
            x=torch.tensor([math.log(ratio)],dtype=torch.float64,requires_grad=True)
            online_loss(x,torch.zeros_like(x),adv).backward();self.assertEqual(float(x.grad),0)
    def test_demo_exact_gradient(self):
        x=torch.tensor([math.log(.1)],dtype=torch.float64,requires_grad=True)
        y=demo_loss(x,.5,.1);y.backward()
        self.assertAlmostEqual(float(y.detach()),-.25,12);self.assertAlmostEqual(float(x.grad),-.125,12)
    def test_zero_demo_gradient(self):
        x=torch.tensor([-1.,-2.],requires_grad=True);demo_loss(x,0).backward();self.assertEqual(float(x.grad.abs().sum()),0)
    def test_finite_low_probability(self):
        x=torch.tensor([-1000.,-100.,-.001],requires_grad=True);loss=demo_loss(x,.75);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(x.grad).all())
if __name__=="__main__":unittest.main()
