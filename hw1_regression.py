import numpy as np
import sys

lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

print(X_train.shape)

## Solution for Part 1
def part1():
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    lambda_mat = lambda_input*np.identity(X_train.shape[1])
    X_transpose = np.transpose(X_train)
    X_prod = np.matmul(X_transpose, X_train)
    first_term = np.linalg.matrix_power(lambda_mat + X_prod, -1)
    sec_term = np.matmul(first_term, X_transpose)
    result = np.matmul(sec_term, y_train)
    return result

wRR = part1()  # Assuming wRR is returned from the function
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file


## Solution for Part 2
def part2():
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    pass

active = part2()  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file
