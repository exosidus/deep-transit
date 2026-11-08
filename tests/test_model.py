import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class CNNTests(unittest.TestCase):
    def test_architecture_returns_one_logit_per_sample(self):
        import torch

        from models.cnn import DeepTransitCNN

        model = DeepTransitCNN().eval()
        with torch.inference_mode():
            output = model(torch.zeros(2, 1, 1000))
        self.assertEqual(tuple(output.shape), (2, 1))

    def test_rejects_wrong_input_shape(self):
        import torch

        from models.cnn import DeepTransitCNN

        with self.assertRaisesRegex(ValueError, "Expected input shape"):
            DeepTransitCNN()(torch.zeros(1, 1, 999))


if __name__ == "__main__":
    unittest.main()
