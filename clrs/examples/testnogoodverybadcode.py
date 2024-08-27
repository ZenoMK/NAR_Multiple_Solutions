import unittest

listOfProb = [
    [[0.2,0.8],
    [0.1,0.9]],

    [[0.2,0.8],
    [0.1,0.9]]
]

class DataPointDummy:
    def __init__(self,data):
        self.data = data

dict = {
    'pi': DataPointDummy(listOfProb)
}

listofdict = [dict, dict]
listofdp = [DataPointDummy(listOfProb)]

##########################################
# graph: [0,0,0]
test_0s = [[1,0,0],
           [1,0,0],
           [1,0,0]]

test_bad = [[0.3,0.3,0.4],
            [0.4,0.3,0.3],
            [0.3,0.4,0.3]]

test_diamond = [[1,0,0,0],
                [1,0,0,0],
                [1,0,0,0],
                [0,0.5,0.5,0]]

expect_diamond = [[0,0,0,1], [0,0,0,2]]

test_shortcut = [[1,0,0,0,0],
                 [1,0,0,0,0],
                 [1,0,0,0,0],
                 [0,0,1,0,0],
                 [0,0.5,0.25,0.25,0]]

expect_shortcut = [[0,0,0,2,3], [0,0,0,2,2], [0,0,0,2,1]]

test_binary_tree = [[1,0,0,0,0,0,0],
                    [1,0,0,0,0,0,0],
                    [1,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0],
                    [0,1,0,0,0,0,0],
                    [0,0,1,0,0,0,0],
                    [0,0,1,0,0,0,0]]

expect_binary_tree = [0,0,0,1,1,2,2]

test_tricky = [[1,0,0]]


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
