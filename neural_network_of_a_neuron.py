import math

input = 0
output_desire = 1

input_weight = 0.5

learning_rate = 0.1

def activation(sum):
    if sum >= 0:
        return 1
    else:
        return 0

print("input: ", input, " desire: ", output_desire)

error = math.inf
iteration = 1

# If the input is equals 0, then the programing will broken. To evited, is possible create a 'bias' (virtual neuron) with static value
bias = 1
bias_weight = 0.5

while not error == 0:
    
    print(" --- iteration: ", iteration)
    iteration = iteration + 1
    print("weight: ", input_weight)

    sum = (input * input_weight) + (bias + bias_weight)

    output = activation(sum)
    print("output: ", output)

    error = output_desire - output
    print("error: ", error)

    if not error == 0:
        input_weight = input_weight + (learning_rate * input * error)
        bias_weight = bias_weight + (learning_rate * bias * error)

print("CONGRATULATIONS!!! THE NETWORK LEARNED")
