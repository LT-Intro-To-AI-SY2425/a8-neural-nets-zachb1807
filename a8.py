from neural import *

# part 1 code
nn = NeuralNet(2, 8, 1)
xor_training_data = [
      # input      # output
    ([0.0, 0.0],    [0.0]), # [0, 0] => 0
    ([0.0, 1.0],    [1.0]), # [0, 1] => 1
    ([1.0, 0.0],    [1.0]), # [1, 1] => 1
    ([1.0, 1.0],    [0.0])  # [1, 0] => 0
]

# nn.train(xor_training_data, iters=10000)

# print([expected[0] - actual[0] for _, expected, actual in nn.test_with_expected(xor_training_data)])


# voter opinions
voternet = NeuralNet(5, 8, 1)
voter_training_data = [
      # input      # output
    ([0.9, 0.6, 0.8, 0.3, 0.1],    [1.0]), # [0, 0] => 0
    ([0.8, 0.8, 0.4, 0.6, 0.4],    [1.0]), # [0, 0] => 0
    ([0.7, 0.2, 0.4, 0.6, 0.3],    [1.0]), # [0, 0] => 0
    ([0.5, 0.5, 0.8, 0.5, 0.8],    [0.0]), # [0, 0] => 0
    ([0.3, 0.1, 0.6, 0.8, 0.8],    [0.0]), # [0, 0] => 0
    ([0.7, 0.3, 0.4, 0.3, 0.6],    [0.0]), # [0, 0] => 0
]

voter_testing_data = [
      # input      # output
    ([1.0, 1.0, 1.0, 0.1, 0.1]), # [0, 0] => 0
    ([0.5, 0.2, 0.1, 0.7, 0.7]), # [0, 0] => 0
    ([0.8, 0.3, 0.3, 0.3, 0.8]), # [0, 0] => 0
    ([0.8, 0.3, 0.3, 0.8, 0.3]), # [0, 0] => 0
    ([0.9, 0.8, 0.8, 0.3, 0.6]), # [0, 0] => 0
]

voternet.train(voter_training_data, iters=10000)


print(voternet.test(voter_testing_data))