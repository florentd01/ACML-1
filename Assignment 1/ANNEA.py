import math

import numpy as np
import random
import matplotlib.pyplot as plt
import main as m


def relu(x):
    return np.maximum(0, x)


# Relu derivative, if x > 0 then 1, else 0
def reluderivative(x):
    return 1. * (x > 0)


# Sigmoid Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Leaky Relu activation function
def leaky_relu(x):
    return np.maximum(0.01 * x, x)


# Leaky Relu derivative, if x < 0, give 0.01, if x>=0 give 1
def leaky_relu_derivative(x):
    returnvalues = x.copy()
    returnvalues[returnvalues < 0] = 0.01
    returnvalues[returnvalues >= 0] = 1
    return returnvalues


# Forward propagate using leaky_relu activation function
def forwardpropleaky(W1, W2, X):
    O1 = leaky_relu(np.dot(W1, X))
    O2 = np.round(np.dot(W2, O1))

    return O1, O2


# Forward prop using relu function
def forwardprop(W1, W2, X):
    O1 = relu(np.dot(W1, X))
    O2 = np.round(np.dot(W2, O1))

    return O1, O2


# Cost function using L2 norm
def cost(ypred, y):
    return (1 / 2) * (np.sum(y - ypred) ** 2)


def getFitness(individual):
    return individual.fitness

# JEAN

class Individual:
    def __init__(self, W1, W2):
        self.W1 = W1
        self.W2 = W2
        self.fitness = -1

    def update_fitness(self, other_fitness):
        # O1, O2 = forwardprop(self.W1, self.W2, input)
        new_fitness = m.run_game(weights=[self.W1, self.W2], scr=False, other_fitness=other_fitness)
        # if (self.fitness != new_fitness):
        #     print("bruh moment hey peter bruh")
        self.fitness = new_fitness
        # self.fitness = abs(np.sum(np.array([1,1] - O2)))

    def getFitness(self):
        return self.fitness


# JEAN
class Population:
    # TODO do different initialization

    def __init__(self, population_size=100, function=True, random_init=True, squared=False, input_layer_size=12,
                 hidden_layer_size=4, output_layer_size=2):
        self.population = []
        self.squared = squared
        for i in range(population_size):
            temp_x = np.random.normal(scale=0.5, size=(hidden_layer_size, input_layer_size + hidden_layer_size))
            temp_y = np.random.normal(scale=0.5, size=(output_layer_size, hidden_layer_size))
            self.population.append(Individual(temp_x, temp_y))

    def update_all(self, input, other_fitness):
        for i in range(len(self.population)):
            self.population[i].update_fitness(other_fitness)

    def elite_update_population(self, X):
        self.population.sort(key=getFitness, reverse=True)
        next_population = self.elite_selection()
        self.population = next_population

    def tour_update_population(self, mutation_chance=0.05):
        self.population.sort(key=getFitness, reverse=False)
        next_population = self.population[:5]

        while len(next_population) < len(self.population):

            ind1, ind2 = self.tournament_selection()
            temp_ind = self.crossover(ind1, ind2)

            # CHANCE FOR MUTATION
            if random.uniform(0, 1) < mutation_chance:
                temp_ind = self.mutation(temp_ind)

            next_population.append(temp_ind)
        self.population = next_population

    def mutation(self, genome):
        # TODO create mutation function
        x = genome.W1.shape[0]
        y = genome.W1.shape[1]

        for i in range(x):
            for j in range(y):
                if random.random() < 0.05:
                    genome.W1[i][j] = random.random()

        x = genome.W2.shape[0]
        y = genome.W2.shape[1]

        for i in range(x):
            for j in range(y):
                if random.random() < 0.05:
                    genome.W2[i][j] = random.random()

        return genome
    # JAN
    def other_mutation(self, genome, mutation_intensity=0.1):
        x = genome.W1.shape[0]
        y = genome.W1.shape[1]

        for i in range(x):
            for j in range(y):
                if random.random() < 0.7:
                    if random.random() < 0.5:
                        genome.W1[i][j] = (1 + mutation_intensity) * genome.W1[i][j]
                    else:
                        genome.W1[i][j] = (1 - mutation_intensity) * genome.W1[i][j]

        x = genome.W2.shape[0]
        y = genome.W2.shape[1]

        for i in range(x):
            for j in range(y):
                if random.random() < 0.7:
                    if random.random() < 0.5:
                        genome.W2[i][j] = (1 + mutation_intensity) * genome.W2[i][j]
                    else:
                        genome.W2[i][j] = (1 - mutation_intensity) * genome.W2[i][j]

        return genome
        # JEAN

    def crossover(self, a, b):
        # TODO create crossover function
        x = a.W1.shape[0]
        y = a.W1.shape[1]

        crossX = random.randint(0, x)
        crossY = random.randint(0, y)

        newAW1 = np.zeros((x, y))
        newBW1 = np.zeros((x, y))

        for i in range(x):
            for j in range(y):
                if i >= crossX and j >= crossY:
                    newAW1[i][j] = b.W1[i][j]
                    newBW1[i][j] = a.W1[i][j]
                else:
                    newAW1[i][j] = a.W1[i][j]
                    newBW1[i][j] = b.W1[i][j]

        x = a.W2.shape[0]
        y = a.W2.shape[1]

        newAW2 = np.zeros((x, y))
        newBW2 = np.zeros((x, y))

        crossX = random.randint(0, x)
        crossY = random.randint(0, y)

        for i in range(x):
            for j in range(y):
                if i >= crossX and j >= crossY:
                    newAW2[i][j] = b.W2[i][j]
                    newBW2[i][j] = a.W2[i][j]
                else:
                    newAW2[i][j] = a.W2[i][j]
                    newBW2[i][j] = b.W2[i][j]

        newA = Individual(newAW1, newAW2)

        newB = Individual(newBW1, newBW2)

        return newA, newB

    def elite_selection(self, top_g=2, mut_prob=0.3, normalCrossover=False, other_mutation=False):
        new_pop = []
        self.population.sort(key=getFitness, reverse=True)
        # print("top g's")
        for i in range(top_g):
            # print(self.population[i].fitness)
            new_pop.append(self.population[i])

        for i in range((len(self.population) - top_g) // 2):
            dad = self.population[i * 2]
            mum = self.population[i * 2 + 1]

            son, daughter = self.crossover(dad, mum)

            # son, daughter = dad, mum

            if random.random() >= mut_prob:
                new_pop.append(son)
            else:
                if other_mutation:
                    new_pop.append(self.other_mutation(son))

                else:
                    new_pop.append(self.mutation(son))

            if random.random() >= mut_prob:
                new_pop.append(daughter)
            else:
                if other_mutation:
                    new_pop.append(self.other_mutation(daughter))

                else:
                    new_pop.append(self.mutation(daughter))

        return new_pop

    def tournament_selection(self, k=5, r=10):

        self.population.sort(key=getFitness, reverse=False)

        indices = random.sample(range(len(self.population)), k)

        indices.sort()

        return self.population[indices[0]], self.population[indices[1]]


# JEAN
def distance_metric(ind1, ind2):
    dist1 = np.linalg.norm(ind2.W1 - ind1.W1)
    dist2 = np.linalg.norm(ind2.W2 - ind1.W2)
    return dist1 + dist2


def main():
    # REMCO
    ## WEIGHTS FOR DIFFERENT BOTS
    # W1_Pop =  [[-0.37284436,  0.29743931,  0.72483957,  0.29821715,  0.33566998,
    #    -0.81331421, -0.38072625,  0.42635615,  0.56533055, -0.76675898,
    #     0.46878375, -0.60725645,  0.66412566, -0.6775803 ,  0.3082418 ,
    #     0.09753266],[ 0.96644359,  0.71250825,  0.54587431,  0.26759514,  0.77262209,
    #     0.75018169,  0.79219529,  0.56555215,  0.06964835,  0.68739279,
    #     0.60925339,  0.13560933,  0.58525759,  0.30563612, -0.12273837,
    #     0.98488623],[ 0.24630054,  0.90978069,  0.77185845,  0.20196556,  0.15825404,
    #     0.55535858, -0.08188738,  0.14950386,  0.04919221,  0.86462799,
    #     0.04665966,  0.85703494,  0.01833025,  0.08915531,  0.12907026,
    #     0.51559982],[0.18852516, 0.18130867, 0.40724674, 0.45758113, 0.10665522,
    #    0.09072679, 0.07111735, 0.22698946, 0.1940678 , 0.05969631,
    #    0.2047378 , 0.92310823, 0.00682807, 0.05466889, 0.25557394,
    #    0.10183195]]
    # W2_Pop = [[0.64265775, 0.12125349, 0.38168601, 0.28939721],[0.3331055 , 0.78351528, 0.60198452, 0.00904068]]

    # W1_Pop =  [[-0.37284436,  0.29743931,  0.72483957,  0.29821715,  0.33566998,
    #    -0.81331421, -0.38072625,  0.42635615,  0.56533055, -0.76675898,
    #     0.46878375, -0.60725645,  0.66412566, -0.6775803 ,  0.3082418 ,
    #     0.09753266],[ 0.96644359,  0.71250825,  0.54587431,  0.26759514,  0.77262209,
    #     0.75018169,  0.79219529,  0.56555215,  0.06964835,  0.68739279,
    #     0.60925339,  0.13560933,  0.58525759,  0.30563612, -0.12273837,
    #     0.98488623],[ 0.24630054,  0.90978069,  0.77185845,  0.20196556,  0.15825404,
    #     0.55535858, -0.08188738,  0.14950386,  0.04919221,  0.86462799,
    #     0.04665966,  0.85703494,  0.01833025,  0.08915531,  0.12907026,
    #     0.51559982],[0.18852516, 0.18130867, 0.40724674, 0.45758113, 0.10665522,
    #    0.09072679, 0.07111735, 0.22698946, 0.1940678 , 0.05969631,
    #    0.2047378 , 0.92310823, 0.00682807, 0.05466889, 0.25557394,
    #    0.10183195]]
    # W2_Pop = [[0.64265775, 0.12125349, 0.38168601, 0.28939721],[0.3331055 , 0.78351528, 0.60198452, 0.00904068]]


    # BEST SIMPLE ROOM COLLISION ONLY
    # W1_Pop = [[ 0.52830065,  0.82026467, -0.60239509,  0.70490659, -0.79455774,
    #    -0.27609968, -0.90440266,  0.22819121, -0.02795927, -0.0723324 ,
    #     0.18419476,  0.5092995 ,  0.67252458, -0.77815817,  0.27216893,
    #    -0.65888525],[-0.34923867, -0.32267543,  0.22176586, -0.91786419,  0.31030124,
    #     0.31737734, -0.43975113, -0.21121485,  0.33472963, -0.24646858,
    #    -0.53312424, -0.73716686,  0.22067111,  0.54955169,  0.17106031,
    #     0.06649181],[-0.11984097, -0.03295657, -0.10652002,  0.95276351,  0.32804639,
    #     0.50314631,  0.03719997,  0.28450611, -0.30949156, -0.48373682,
    #     1.23500255,  0.13976125, -0.41126059, -0.56879167, -0.02234095,
    #    -0.10466104],[-0.4217718 , -0.48515979, -0.06256681, -0.72837172, -0.49851259,
    #     0.56049555,  1.11033025, -0.30373323, -0.0732397 , -0.22026435,
    #     0.04293587,  0.39678678, -0.85386897,  0.16797849, -0.57144017,
    #     0.48741062]]
    # W2_Pop = [[0.16065984, 0.36730401, 0.81416337, 0.16222643],[ 0.60923082,  0.09315206,  0.36040098, -0.63374744]]


    # BEST SIMPLE ROOM COLLISION AND AREA
    # W1_Pop = [[-0.13427526,  0.53190671,  0.69260607,  0.63090453, -0.06825471,
    #    -0.10397452, -0.49307912,  0.50449873,  0.26861121,  0.55140469,
    #     0.14456538,  0.31144976,  0.43986051,  0.07883826, -0.25478465,
    #    -0.41938765],[ 0.01729253,  0.17770436,  0.50330436, -0.30099999, -0.24302607,
    #    -0.67843552,  0.42540896, -0.07209495, -0.63822179, -0.24296894,
    #    -0.14144233,  0.08735752, -1.05502374, -0.90181943,  0.64884928,
    #    -0.44036572],[ 0.39183856, -0.5932075 , -0.18858308,  0.16970817, -0.24256502,
    #     0.1068958 , -0.83855107,  0.08953281, -0.56257163, -0.05863133,
    #    -0.82737578, -0.95172003,  0.02358235, -0.51884131, -0.93428401,
    #    -1.07441391],[ 0.12081814,  0.22292204,  0.04974228, -0.21143455,  0.05237925,
    #     0.39826351, -0.43287296,  0.26500996, -0.07316552, -0.43215978,
    #    -0.437759  ,  0.31865409,  0.46007971, -0.32068083,  1.23804843,
    #    -0.49252923]]
    # W2_Pop = [[ 0.86300512, -0.69835988,  0.3011599 ,  0.26305234],[ 0.34449345,  0.47352822, -0.46257123,  0.39764071]]

    # BEST HARDER ROOM COLLISION ONLY
    # W1_Pop = [[ 0.53867998,  0.01726782,  0.87556051, -0.37399934,  0.17806312,
    #    -0.39686549, -0.47165092,  0.28456108,  0.09034143,  0.71816886,
    #    -0.13750441, -0.05331679, -0.41476447,  0.61475378,  0.29471077,
    #     0.196594  ],[-0.31566137,  0.66352259, -0.29022446,  0.57700851,  0.19616531,
    #     0.3165126 ,  0.26613422, -0.21611477,  0.40334083,  0.93729172,
    #    -0.53424248,  0.87998355,  0.18449758,  0.15461054,  0.45864133,
    #     0.35180956],[-1.28481969, -0.35361896, -0.27271653,  0.14115755,  0.03250513,
    #     0.09530747,  0.10238649,  0.18097107,  0.57739571, -0.15747655,
    #     0.34081727,  0.42050128,  0.70153283, -0.33926616,  0.45561423,
    #     0.96517829],[ 0.15238752, -0.42841987,  0.31218957, -0.17992386, -0.07828701,
    #     1.01305364,  0.37565042,  0.1517493 ,  0.8661942 ,  0.22534674,
    #    -0.41961335,  0.0143424 , -0.1424979 ,  0.68320046, -0.58128675,
    #     0.60884916]]
    #
    # W2_Pop = [[-0.24945845,  0.64909111,  1.00210402,  0.42772294],[0.94736035, -0.01961426, -0.21745735,  0.25172618]]
    # INTERMEDIATE HARDER ROOM
    # W1_Pop = [[-0.09293194, -0.07011743,  0.56814603, -0.61739308,  0.90253403,
    #     0.4825991 ,  0.34853305,  0.24336759, -0.10339635,  0.37780948,
    #    -0.35620307,  0.45062784,  0.2466962 ,  0.22975557, -0.30390931,
    #     0.33694423],[-0.04664667,  0.14689359, -0.09616835, -0.53221598,  0.42296254,
    #     0.0914473 , -0.99574536, -0.28120516, -0.28451067,  0.26171452,
    #    -0.39500581,  0.17190074, -0.8946769 , -0.12180884, -0.37761632,
    #     0.31678058],[ 0.14970233,  0.17737509,  0.55485285, -0.33752647,  0.82332299,
    #    -0.64188012,  0.03591596,  0.37068991,  0.14626553, -0.41108393,
    #     0.00863465,  0.50763216,  0.09097544,  0.22819123, -0.23993467,
    #    -0.01362642],[-0.87718085, -0.89299811,  0.62528531,  0.29029023,  0.46291507,
    #    -0.26073099,  0.02866133,  0.85077798,  0.81104216, -0.55514774,
    #    -0.16590487, -0.09709294,  0.0548146 , -0.21818712, -0.00106874,
    #    -0.17642381]]
    # W2_Pop = [[0.35439666, -0.16762937, 0.03475125, -0.56064929],[-0.43631938, 0.12292398, 0.88445215, 0.21253978]]
    #


    # BEST COLLISION AND AREA
    W1_Pop = [[ 0.30345912, -0.68535237,  0.22042024, -0.26850724, -0.43337223,
       -0.8907365 , -0.23820837, -0.08037018, -0.10653413,  0.42034419,
        0.25473331,  0.56247203,  1.65532557, -0.46427056,  0.76277987,
        1.47563339],[ 0.23334905, -0.80734499, -0.43411001,  0.47608518, -0.23878705,
        0.52866228, -0.01515896,  0.23018678, -0.29987509, -0.05068551,
        0.71990481, -0.26856787, -0.26675368,  0.41807527,  0.01110528,
        0.14051577],[ 0.68250445,  0.032039  ,  0.71247333, -0.78466513,  0.55212851,
       -0.11296878,  0.08073898,  0.13577679,  0.02742289, -0.35358715,
        0.66975992,  0.20548088, -0.28607285, -0.43732328,  0.296511  ,
        0.17825981],[-0.6238172 , -0.6484651 ,  0.32133305,  0.0659446 ,  0.46056348,
        0.45687478,  0.30289232,  0.49150708, -0.16301399,  0.12116777,
        0.32521239,  0.57683496,  0.25367389, -0.21307737,  0.39663789,
        0.21272589]]
    W2_Pop = [[0.90597379, 0.84838231, 0.58830717, 0.45323027],[-1.00911451, -0.1040837 ,  1.09127664,  0.4932841 ]]


    # Intermediate COLLSION AND AREA
    # W1_Pop = [[-0.30059726, -0.15339281, -0.17392629, -0.07774291, -0.57337714,
    #    -0.45180188,  0.30567281,  0.23111639,  0.47768581, -0.22570414,
    #     0.59648513,  0.79992452, -0.98148445, -0.11963576,  0.37773496,
    #     0.43619433],[ 0.15302916, -0.34536737, -0.05264681,  0.49260568, -0.33407964,
    #     0.80977952,  0.54568893,  0.73561465, -0.24658266, -1.30347088,
    #     0.06455353,  0.31348907, -0.82791757,  0.75620484,  0.96221531,
    #    -0.07615206],[-0.14944352, -0.62936382, -0.18060333,  0.27088607, -0.10004487,
    #    -0.15308581, -0.6382473 , -0.40709311, -0.29631336, -0.4752317 ,
    #    -0.06125926, -0.4499828 , -0.02775835, -0.47519166, -0.50078156,
    #     0.03967032],[ 0.71986105,  0.66807974,  0.01973496, -1.04768467,  0.04894896,
    #    -0.51649296,  0.88425232,  0.49150708, -0.85991442, -0.03967601,
    #     0.22817014,  0.48950108, -0.21707288,  0.48467792,  0.33009332,
    #     0.00401854]]
    # W2_Pop = [[-0.41075453,  0.84838231,  0.58830717,  0.45323027],[0.21718471, 0.16410291, 0.14519394, 0.27030991]]


    # LAST
    # W1_Pop = [[ 3.01107963e-01,  9.76796865e-01,  1.35118988e+00, -1.26992070e-03,
    #     1.97592373e-01, -3.18178045e-01,  1.42922922e+00,  6.23557794e-01,
    #     1.79508916e-01,  3.78821608e-01, -4.03631304e-01, -2.43210666e-01,
    #     2.13756222e-01,  9.02024045e-01,  4.46325177e-01,  1.23425061e-01],[-0.34208987, -0.09226063,  0.06914116,  0.39806907,  0.14061811,
    #     0.57078301,  0.8987314 ,  0.94231934,  0.30701394, -1.14466568,
    #     0.11179007, -0.03850057,  0.63563141, -0.92692506, -0.16840494,
    #    -0.20580472],[ 0.17050394,  1.04292554, -1.46419177, -0.0422919 , -0.27881027,
    #     0.30241326,  0.27672086, -0.25868788, -0.50310113,  0.01804631,
    #     0.33173574, -0.43584541,  0.29852783,  0.42384491,  0.51398696,
    #     0.3042026 ],[ 0.16628524, -0.33723818,  0.32340158,  0.06251068,  0.07572263,
    #    -0.741889  , -0.10333859,  0.64576078, -0.44518805,  0.72021596,
    #     0.21261029,  0.40640156,  0.17256384,  0.16013114,  0.36931255,
    #    -0.43872441]]
    # W2_Pop = [[ 0.50911424, -0.12589808,  0.17528495,  0.68805778],[-0.16112987,  1.17354354,  0.95197377, -0.64130113]]

    # RAndom bot
    W1_Pop = np.random.randn(4, 16)
    W2_Pop = np.random.randn(2, 4)

    m.run_game(weights= [W1_Pop, W2_Pop], scr=True, max_timesteps=2000)
    # exit(0)
    # m.run_game(scr=True, max_timesteps=10000000)
    no_gens = 30
    # Experiment Population size
    pop_sizes = [10, 50]
    pop_size = 100
    # Experiment Initalization
    random_init = True

    # Experiment squared fitness function, false means just the amount cleaned
    other_fitness = True

    # Experiment with Selection and reproduction
    tournament = False

    # Experiment Crossover
    normalCrossover = False

    # Experiment mutation
    mutation_chance = 0.7

    # Stop criterion, already encapsulated in function
    stop_after = 1000

    # To experiment with the different input training data, remove certain datapoints here for X and Y
    # our X data
    # Doesn't get used in function itself
    X = np.array([200, 50, 73, 157, 200, 200, 100, 0, 56, 23, 40, 150])

    # number of runs
    no_runs = 1

    best_fitness_list = []
    avg_fitness_list = []
    diversity_list = []
    # JAN & REMCO
    for i in range(no_runs):
        # Best list fitness
        best_fitness = []

        # population average
        avg_fitness = []

        # diversity list
        diversity = []

        pop = Population(population_size=pop_size)
        prev_fitness = 0
        prev_weights = [3]
        # number of generations
        for i in range(no_gens):
            print("Generation: {}".format(i))
            # Perform all fitnesses and decide which fitness to use
            pop.update_all(X, other_fitness=other_fitness)
            pop.population.sort(key=getFitness, reverse=True)

            if pop.population[0].fitness < prev_fitness:
                # if prev_weights != pop.population[0].W1:
                print("test")
                print(prev_fitness)
                print(pop.population[0].fitness)
                print("WEIGHTS1")
                print(list(pop.population[0].W1))
                print("WEIGHTS1_test")
                print(list(prev_weights))
                # print("WEIGHTS2")
                # print(list(pop.population[0].W2))
            prev_weights = pop.population[0].W1
            prev_fitness = pop.population[0].fitness

            print(pop.population[0].fitness)
            print("WEIGHTS1")
            print(list(pop.population[0].W1))
            print("WEIGHTS2")
            print(list(pop.population[0].W2))

            # add best performing to list and average
            best_fitness.append(pop.population[0].fitness)
            sum = 0
            for ind in pop.population:
                sum += ind.fitness
            avg_fitness.append(sum / len(pop.population))

            # Calculate diversity
            sum_diversity = 0
            for i in range(len(pop.population)):
                for j in range(len(pop.population)):
                    sum_diversity += distance_metric(pop.population[i], pop.population[j])
            diversity.append(sum_diversity)

            # Use selection
            pop.elite_update_population(X)

        best_fitness_list.append(best_fitness)
        avg_fitness_list.append(avg_fitness)
        diversity_list.append(diversity)

    se_best_fitness = np.std(best_fitness_list, ddof=1, axis=0) / np.sqrt(np.size(best_fitness_list, axis=0))
    average_best_fitness = np.mean(best_fitness_list, axis=0)
    plt.errorbar(range(no_gens), average_best_fitness, yerr=se_best_fitness, linestyle='None', marker='.')
    plt.xlabel("Number of generation")
    plt.ylabel("Fitness Score")
    plt.title("Best Fitness")
    plt.show()

    se_avg_fitness = np.std(avg_fitness_list, ddof=1, axis=0) / np.sqrt(np.size(avg_fitness_list, axis=0))
    average_avg_fitness = np.mean(avg_fitness_list, axis=0)
    plt.errorbar(range(no_gens), average_avg_fitness, yerr=se_avg_fitness, linestyle='None', marker='.')
    plt.xlabel("Number of generation")
    plt.ylabel("Fitness Score")
    plt.title("Average Fitness")

    plt.show()

    se_diversity = np.std(diversity_list, ddof=1, axis=0) / np.sqrt(np.size(diversity_list, axis=0))
    average_diversity = np.mean(diversity_list, axis=0)
    plt.errorbar(range(no_gens), average_diversity, yerr=se_diversity, linestyle='None', marker='.')
    plt.xlabel("Number of generation")
    plt.ylabel("Diversity Score")
    plt.title("Diversity")

    plt.show()


if __name__ == "__main__":
    main()
