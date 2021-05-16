import torch
from transformer import Transformer
import unittest

class TransformerTest(unittest.TestCase):
    def test_main(self):
        src = torch.rand(64, 16, 512)
        tgt = torch.rand(64, 16, 512)
        out = Transformer()(src, tgt)
        print(out.shape)
        # torch.Size([64, 16, 512])

if __name__=='__main__':
    unittest.main()