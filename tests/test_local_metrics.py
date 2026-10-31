import unittest
import numpy as np
import torch

from local_metrics import frechet_distance, knn_radii, precision_recall


class LocalMetricsTests(unittest.TestCase):
    def test_gaussian_distance_identity_translation_and_symmetry(self):
        x = np.random.default_rng(12).normal(size=(200, 8))
        self.assertAlmostEqual(frechet_distance(x, x), 0, places=6)
        self.assertAlmostEqual(frechet_distance(x, x+2), 8*4, places=6)
        y = x * 0.4 + 1
        self.assertAlmostEqual(frechet_distance(x,y),frechet_distance(y,x),places=6)

    def test_singular_features(self):
        x = np.ones((20, 8))
        self.assertEqual(frechet_distance(x,x),0)
        self.assertEqual(frechet_distance(x,x+1),8)

    def test_knn_excludes_self_and_matches_brute_force(self):
        x = torch.tensor([[0.],[1.],[3.],[7.]],dtype=torch.float64)
        radius = knn_radii(x,k=1,chunk_size=2)
        self.assertTrue(torch.equal(radius,torch.tensor([1.,1.,4.,16.],dtype=torch.float64)))
        expected = torch.cdist(x,x).square()
        expected.fill_diagonal_(torch.inf)
        self.assertTrue(torch.allclose(knn_radii(x,k=2,chunk_size=3),expected.kthvalue(2,dim=1).values))

    def test_manifolds_identity_and_separation(self):
        x = torch.tensor([[0.],[1.],[2.],[3.],[4.]],dtype=torch.float64)
        self.assertEqual(precision_recall(x,x,k=1,chunk_size=2),{'precision':1.,'recall':1.})
        self.assertEqual(precision_recall(x,x+100,k=1,chunk_size=3),{'precision':0.,'recall':0.})

    def test_chunked_membership_matches_brute_force(self):
        rng = np.random.default_rng(4)
        x,y = torch.tensor(rng.normal(size=(21,4))),torch.tensor(rng.normal(size=(18,4)))
        rr,rf = knn_radii(x,3,5),knn_radii(y,3,4)
        distances = torch.cdist(y,x).square()
        expected = {'precision':(distances<=rr[None,:]).any(1).double().mean().item(),
                    'recall':(distances<=rf[:,None]).any(0).double().mean().item()}
        actual = precision_recall(x,y,3,7)
        for key in expected:
            self.assertAlmostEqual(expected[key],actual[key])


if __name__ == '__main__':
    unittest.main()
