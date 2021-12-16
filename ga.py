from itertools import product
import pandas as pd
from random import choice, choices, randint, sample, seed, uniform
from statistics import mean
from string import ascii_uppercase

# Berghman genetic algorithm

seed(42)

class GeneticAlgorithm:
    def __init__(self, colors, user_code):
        self.N = 150
        self.max_gen = 100
        self.code = user_code
        self.colors = colors
        self.places = len(user_code)
        self.options = [list(tup) for tup in product(ascii_uppercase[:self.colors], repeat=self.places)]
        self.p_cross_over = 0.5
        self.p_mutation = 0.03
        self.p_permutation = 0.03
        self.p_inversion = 0.02
        self.guesses = {} # {(guess): (black, white)}
        self.max_e_size = 60 # max size of eligible combinations
        self.eligible_combinations = set([])

    def create_population(self):
        return sample(self.options, self.N)

    def fitness(self, individual, a=1, b=2):
        xs, ys = 0, 0
        for guess in self.guesses.keys():
            (Xp, Yp) = general_response(guess, individual)
            xs += abs(Xp - self.guesses[guess][0])
            ys += abs(Yp - self.guesses[guess][1])
        if xs == 0 and ys == 0:
            self.eligible_combinations.add(tuple(individual))
        return (a * xs) + ys + (b * self.places * (len(self.guesses) - 1))

    def selection(self, population):
        weights_list = [self.fitness(ind) for ind in population]
        parents = [list(choices(population, weights=weights_list, k=2)) for i in range(int(self.N/2))]
        return parents

    def cross_over(self, parent1, parent2):
        child1 = parent1.copy()
        child2 = parent2.copy()
        if uniform(0, 1) <= self.p_cross_over:
            p1 = randint(0, self.places-1)
            p2 = randint(0, self.places-1)
            if p2 < p1:
                tmp = p1
                p1 = p2
                p2 = tmp
            if p1 != p2:
                tmp = child1[p1 + 1:p2].copy()
                child1[p1 + 1:p2] = child2[p1 + 1:p2]
                child2[p1 + 1:p2] = tmp
                if len(tmp) == 0:
                    tmp1 = child1[p2 + 1:] + child1[:p1]
                    tmp2 = child2[p2 + 1:] + child2[:p1]
                    p_l = len(child1[p2 + 1:])
                    child1[p2 + 1:] = tmp2[:p_l]
                    child1[:p1] = tmp2[p_l:]
                    child2[p2 + 1:] = tmp1[:p_l]
                    child2[:p1] = tmp1[p_l:]
            elif p1 == p2:
                tmp = child1[p1 + 1:]
                child1[p1 + 1:] = child2[p1 + 1:]
                child2[p1 + 1:] = tmp
        return [child1, child2]

    def mutate(self, individual):
        new_individual = individual.copy()
        if uniform(0, 1) <= self.p_mutation:
            p = randint(0, self.places-1)
            new_individual[p] = choice(ascii_uppercase[:self.colors])
        return new_individual

    def permutate(self, individual):
        new_individual = individual.copy()
        if uniform(0, 1) <= self.p_permutation:
            p1 = randint(0, self.places-1)
            p2 = randint(0, self.places-1)
            new_individual[p1] = individual[p2]
            new_individual[p2] = individual[p1]
        return new_individual

    def inverse(self, individual):
        new_individual = individual.copy()
        if uniform(0, 1) <= self.p_inversion:
            p1 = randint(0, self.places - 1)
            p2 = randint(0, self.places - 1)
            if p2 < p1:
                tmp = p1
                p1 = p2
                p2 = tmp
            inv = individual[p1 + 1:p2].copy()
            if len(inv) > 0:
                inv.reverse()
                new_individual[p1 + 1:p2] = inv
            elif len(inv) == 0:
                inv = individual[p2 + 1:] + individual[:p1]
                inv.reverse()
                p2_l = len(individual[p2 + 1:])
                new_individual[p2 + 1:] = inv[:p2_l]
                new_individual[:p1] = inv[p2_l:]
        return new_individual

    def select_next_guess(self):
        combinations = list(self.eligible_combinations)
        if len(combinations) == 0:
            print(f'\n\nWARNING!\n{self.guesses} --- {self.code}')
            exit(1)
        next_guess = combinations[0]
        if len(combinations) > 1:
            min_score = self.max_e_size
            # print('Selecting next guess')
            for new_guess in combinations:
                options = combinations.copy()
                options.remove(new_guess)
                scores = []
                for secret_code in options:
                    options.remove(secret_code)
                    (X, Y) = general_response(secret_code, new_guess)
                    score = 0
                    for guess in options:
                        (Xp, Yp) = general_response(new_guess, guess)
                        xs = abs(Xp - X)
                        ys = abs(Yp - Y)
                        if xs == 0 and ys == 0:
                            score += 1
                    scores.append(score)
                new_score = mean(scores)
                # print(f'guess {new_guess} - {new_score} avg options in set')
                if new_score <= min_score:
                    next_guess = new_guess
                    # print(f'next guess - {next_guess}')
                    min_score = new_score
        # print(f'NEXT GUESS - {next_guess}')
        return next_guess

    def play(self):
        guess_nbr = 1
        initial_guess = ('A', 'A', 'B', 'C')
        next_guess = initial_guess
        self.guesses[initial_guess] = general_response(self.code, initial_guess)
        (X, Y) = self.guesses[initial_guess]
        while(X != self.places):
            generation = 1
            self.eligible_combinations = set([])
            population = self.create_population()
            while(generation <= self.max_gen and len(self.eligible_combinations) <= self.max_e_size):
                new_generation = set([])
                # print(f'\nGeneration #{generation}')
                for parents in self.selection(population):
                    children = self.cross_over(parents[0], parents[1])
                    for child in children:
                        start_size = len(new_generation)
                        new_individual = self.mutate(child)
                        new_individual = self.permutate(new_individual)
                        new_individual = self.inverse(new_individual)
                        # print('new_individual', new_individual)
                        new_generation.add(tuple(new_individual))
                        while len(new_generation) == start_size:
                            random_ind = choice(self.options)
                            new_generation.add(tuple(random_ind))
                population = [list(i) for i in new_generation]
                generation += 1
                if generation == self.max_gen and len(self.eligible_combinations) == 0:
                    print('Problems with filling eligible combinations')
                    print(f'- guess #{guess_nbr} in game {self.code}')
                    population = self.create_population()
                    generation = 1

            next_guess = self.select_next_guess()
            guess_nbr += 1
            # print(f'Guess #{guess_nbr} in {generation} generations - {next_guess}')
            self.guesses[next_guess] = general_response(self.code, next_guess)
            (X, Y) = self.guesses[next_guess]
        # print(f'YOU WON! code: {self.code} - {next_guess}')
        # print(f'guess nbr: {guess_nbr}')
        return guess_nbr

def general_response(in_code, guess):
    blacks = [0] * len(in_code)
    whites = [0] * len(in_code)
    idxs = []
    code = list(in_code)
    for idx, c in enumerate(in_code):
        if c == guess[idx]:
            blacks[idx] = 1
            idxs.append(idx)
            code[idx] = 'ZZ'

    for idx, c in enumerate(code):
        if blacks[idx] == 0:
            if guess[idx] in code[0: idx:] + code[idx + 1::]:
                if guess[idx] in code[0: idx:]:
                    fit = code[0: idx:].index(guess[idx])
                    if fit not in idxs:
                        idxs.append(code[0: idx:].index(guess[idx]))
                        whites[idx] = 1
                elif guess[idx] in code[idx + 1::]:
                    fit = idx + code[idx ::].index(guess[idx])
                    if fit not in idxs:
                        idxs.append(idx + code[idx ::].index(guess[idx]))
                        whites[idx] = 1
    return (sum(blacks), sum(whites))


if __name__ == '__main__':
    colors = 6
    places = 4

    import time
    xtimes = 5
    df = pd.DataFrame()
    for idx, code in enumerate(product(ascii_uppercase[:colors], repeat=places)):
        print(idx, code)
        codes, scores, times = [], [], []
        for i in range(xtimes):
            start_time = time.time()
            codes.append(code)
            ga = GeneticAlgorithm(colors, list(code))
            nbr = ga.play()
            stop_time = time.time()
            scores.append(nbr)
            times.append(stop_time-start_time)
        result = pd.DataFrame({'code': codes,
                               'turns': scores,
                               'time': times})
        df = df.append(result, ignore_index=True)

    df.to_csv(f'Berghman_GA_results_x{xtimes}.csv', sep=';')
