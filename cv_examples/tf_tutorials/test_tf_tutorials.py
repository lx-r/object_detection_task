import unittest
from mnist_tutorials.mnist_docs import mnist_test

class Test(unittest.TestCase):
    
    def test_mnist_simple(self):
        mnist_test()

if __name__=="__main__":
    unittest.main()