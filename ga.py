# Genetic algorithm

import numpy
import random
import math
import argparse
import matplotlib.pyplot as pyplot


def objective_function(x):
    return (x**3) - (x**2)


def compute_fitness(x):
    return objective_function(x)


def generate_solutions(n, l):
    # generates n random bitstring solutions of length l
    arr = []
    for i in xrange(0, n):
        y = [random.randint(0, 1) for j in xrange(0, l)]
        arr.append(y)
    
    return arr


def decode_from_binary(bit_string, range):
    # decodes a given bit string from binary to an integer
    # and squeezes the value into a given range
    value = power = 0
    for i in reversed(bit_string):
        if (i == 1):
            value += 2**power
        power += 1
    
    # squeeze value into range
    l = len(bit_string)
    xmin = range[0]
    xmax = range[1]
    squeezed_value = xmin + (float(xmax-xmin)/float((2**l)-1) * float(value))
    
    return int(squeezed_value)


def crossover(solutions, probabilities, bit_length):
    # takes a solutions array and performs crossovers to create new solutions
    # first, uses roulette method to select solutions to mate
    new_solutions = []
    
    # the max survives, then the rest enter the roulette
    for i in xrange(len(solutions)):
        if (new_solutions):
            # random 0-1 for threshold
            threshold = random.uniform(0,1)
            total = 0
            for j, val in enumerate(probabilities):
                total += val
                if (total >= threshold):
                    new_solutions.append(solutions[j])
                    break
        else:
            max = numpy.argmax(probabilities)
            new_solutions.append(solutions[max])
    
    # crossover
    iterations = (len(new_solutions)/2) + 1
    children = []
    
    for i in xrange(1, iterations):
        #print "Iteration # " + str(i)
        a = new_solutions[2 * i - 2]
        b = new_solutions[2 * i - 1]
        
        cut = random.randint(1, bit_length-1)
        
        child_a = a[0:cut] + b[cut:]
        child_b = b[0:cut] + a[cut:]
        
        children.append(child_a)
        children.append(child_b)
    
    return children


def mutate(solutions, mutation_rate):
    # takes a solutions array and mutates elements according to given rate
    for j in solutions:
        for k in xrange(0, len(j)):
            rand = random.randint(0, 100)
            if (rand <= mutation_rate):
                j[k] = 1 - j[k]
    
    return solutions


def init_ga(num_i, num_s, m):
    num_iterations = num_i
    num_solutions = num_s
    # mutation rate given as a %
    mutation_rate = m
    range = (0, 60)
    
    print "Number of iterations: " + str(num_iterations)
    print "Number of solutions: " + str(num_solutions)
    print "Mutation rate: " + str(mutation_rate) + "%\n"
     
    # calculate bit length based on range max
    bit_length = int(math.ceil(math.log(range[1], 2)))
    
    # ensure we have an even number of solutions
    if num_solutions %2 != 0:
        print "Error: number of solutions must be even"
        sys.exit(1)
    
    # generate original solutions
    solutions = generate_solutions(num_solutions, bit_length)
    
    fitnesses = probs = []
    
    # for plotting
    max_values = []
    avg_values = []
    maxf = avgf = 0
    
    for i in xrange(0, num_iterations):
        print "Iteration #" + str(i)
        
        # if first iteration, decode and compute original fitnesses
        if i == 0:
            # decode
            solutions_decoded = [decode_from_binary(j, range) for j in solutions]
            print "Decoded: " + str(solutions_decoded)
            
            # compute fitnesses
            fitnesses = [compute_fitness(j) for j in solutions_decoded]
        
        print "Fitnesses before crossover/mutation: " + str(fitnesses)
        # compute probabilities
        probs = [float(j)/float(sum(fitnesses)) for j in fitnesses]
        print "Probabilities: " + str(probs)
        
        # crossover solutions
        solutions = crossover(solutions, probs, bit_length)
        print "After crossover: " + str(solutions)
        
        # mutate solutions
        solutions = mutate(solutions, mutation_rate)
        print "After mutations: " + str(solutions)
        
        # convert new solutions from binary
        solutions_decoded = [decode_from_binary(j, range) for j in solutions]
        print "Decoded: " + str(solutions_decoded)
        
        # compute new fitnesses
        fitnesses = [compute_fitness(j) for j in solutions_decoded]
        print "New fitnesses: " + str(fitnesses)
        
        maxf = max(fitnesses)
        avgf = numpy.mean(fitnesses)
        max_values.append(maxf)
        avg_values.append(avgf)
        print "Max = " + str(maxf)
        print "Avg = " + str(avgf)
        
        print ""
    
    # plot the graph of max values
    xaxis = xrange(1, num_iterations+1, 1)
    pyplot.xticks([x for x in xaxis])
    pyplot.xlabel("Iteration")
    pyplot.ylabel("Max fitness")
    pyplot.plot(xaxis, max_values)
    pyplot.plot(xaxis, avg_values)
    pyplot.show()


if __name__ == "__main__":
    
    # handle command line arguments
    parser = argparse.ArgumentParser(description="Genetic Algorithm")
    # add arguments for iterations, solutions and mutation rate
    parser.add_argument("-i", "--num_iterations", help="The number of iterations the algorithm will run for; defaults to 3", default=3, type=int)
    parser.add_argument("-s", "--num_solutions", help="The number of solutions; defaults to 4", default=4, type=int)
    parser.add_argument("-m", "--mutation_rate", help="The mutation rate given as a percentage; defaults to 5", default=5, type=int)
    args = parser.parse_args()
    
    # assign command line values to variables
    num_iterations = 0
    num_solutions = 0
    mutation_rate = 0
    if args.num_iterations:
        num_iterations = args.num_iterations
    if args.num_solutions:
        num_solutions = args.num_solutions
    if args.mutation_rate:
        mutation_rate = args.mutation_rate
    
    # run the ga
    init_ga(num_iterations, num_solutions, mutation_rate)
    