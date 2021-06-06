import unittest
from mnist_tutorials import mnist_tutorials

class Test(unittest.TestCase):
    
    def test_mnist_simple():
        mnist_tutorials.mnist_test()

if __name__=="__main__":
    unittest.main()