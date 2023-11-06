from Ann import Ann
import random
from copy import deepcopy

random.seed(1)
def main():
    layers = [8,3,8]
    nets = Ann(layers, 0.9)
    # for layer in nets.weights:
    #     print(layer)
    result = nets.feedforward([0,1,0,0,0,0,0,0])
    print(result)
    result = nets.feedforward([1,0,0,0,0,0,0,0])

    input = [1,0,0,0,0,0,0,0]
    output = deepcopy(input)

    nets.backpropagation(input, output)

if __name__ == "__main__":
    main()