from itertools import product
from random import choices
from string import ascii_uppercase

import numpy as np
import pickle
import time
import os

"""
1. Create the set S of 1296 possible codes (1111, 1112 ... 6665, 6666)
2. Start with initial guess 1122 (Knuth gives examples showing that this algorithm using other first guesses such as 1123, 1234 does not win in five tries on every code)
3. Play the guess to get a response of coloured and white pegs.
4. If the response is four colored pegs, the game is won, the algorithm terminates.
5. Otherwise, remove from S any code that would not give the same response if it (the guess) were the code.
6. Apply minimax technique to find a next guess as follows: 
    For each possible guess, that is, any unused code of the 1296 not just those in S, 
    calculate how many possibilities in S would be eliminated for each possible colored/white peg score. 
    The score of a guess is the minimum number of possibilities it might eliminate from S. 
    A single pass through S for each unused code of the 1296 will provide a hit count 
    for each coloured/white peg score found; 
    the coloured/white peg score with the highest hit count will eliminate the fewest possibilities; 
    calculate the score of a guess by using 
    "minimum eliminated" = "count of elements in S" - (minus) "highest hit count". 
    From the set of guesses with the maximum score, select one as the next guess, choosing a member of S whenever possible. (Knuth follows the convention of choosing the guess with the least numeric value e.g. 2345 is lower than 3456. Knuth also gives an example showing that in some cases no member of S will be among the highest scoring guesses and thus the guess cannot win on the next turn, yet will be necessary to assure a win in five.)
7. Repeat from step 3.
"""

class Game():
    def __init__(self, user_code, colors):
        if check_response_array() == False:
            prepare_response_array(colors, len(user_code))
        self.code = convert_guess(user_code)
        self.colors = colors
        self.places = len(user_code)
        self.general_options = load_options_list()
        self.options_idx = []
        self.guesses = {}
        self.round = 0
        self.code_idx = self.general_options.index(convert_guess(user_code))
        self.responses = load_response_array()
        self.debug = False

    def eliminate_options(self, guess_idx, response):
        x = np.argwhere(self.responses[:, guess_idx] == response)
        r = set(x.flatten())
        r -= self.guesses.keys()
        if self.debug:
            print(f'r - {sorted(r)}')
        if self.round == 1:
            self.options_idx = sorted(r)
        else:
            self.options_idx = [opt for opt in self.options_idx if opt in r]

    def score_counter(self):
        u_responses = np.unique(self.responses)  # unikalne odpowiedzi
        sub_array = self.responses[self.options_idx, :]
        # zliczenie wystąpień każdego score'a w kolumnie
        array = np.empty((len(u_responses), sub_array.shape[1]))
        for idx, response in enumerate(u_responses):
            array[idx, :] = np.count_nonzero(sub_array == response, axis=0)
        return array

    def next_guess(self, guess_idx):
        array = self.score_counter()
        biggest = array.max(axis=0)
        biggest[list(self.guesses.keys())] = len(self.options_idx) # dla zagranych kodów wypełniam max. liczbą
        queue = np.asarray(biggest == min(biggest)).nonzero()[0].tolist()
        queue_x_S = [e for e in queue if e in self.options_idx]

        if len(queue_x_S) > 0:
            next = queue_x_S[0]
        else:
            next = queue[0]

        if self.debug:
            print(f'Rozmiar tablicy: {array.shape}')
            print('SPR:', len(biggest), len(self.options_idx))
            print(biggest)
            print(f'MIN z biggest: {min(biggest)}')
        new_guess = self.general_options[next]
        return new_guess


    def play(self, guess=None):
        if self.round == 0:
            guess = ('A', 'A', 'B', 'B')
        if (len(guess) == len(self.code)) and (set(guess).issubset(set(ascii_uppercase[:self.colors]))):
            self.round += 1
            guess_idx = self.general_options.index(guess)
            response = self.responses[self.code_idx, guess_idx]
            self.guesses[guess_idx] = response
            print(f'Guess: {guess}, response: {response}, round: {self.round}')
            if self.debug:
                print('GUESSES', self.guesses)
            if response != 40:
                self.eliminate_options(guess_idx, response)
                new_guess = self.next_guess(guess_idx) # funkcja do wyznaczenia nowego kandydata
                self.play(new_guess)
            else:
                print(f'YOU WON! code: {self.code}')
        else:
            print(f'error: {guess} is a WRONG GUESS\na guess should have length of {self.places}',
                  f'and be a combination of {ascii_uppercase[:self.colors]}')
            exit(1)


def possibilities(colors, places):
    return list(product(ascii_uppercase[:colors], repeat=places))

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
    return sum(blacks) * 10 + sum(whites)

def convert_guess(guess):
    if isinstance(guess, str):
        return tuple([letter for letter in guess])
    else:
        return guess

def check_response_array():
    files = [f for f in os.listdir()]
    if 'options.pickle' in files and 'response.npy' in files:
        return True
    else:
        return False

def prepare_response_array(colors, places):
    options = possibilities(colors,places)
    A = np.array([[general_response(code, guess) for guess in options] for code in options])
    # code w wierszu, guess w kolumnie [code, guess]
    np.save('response.npy', A)
    pickle.dump(options, open('options.pickle', 'wb'))

def load_response_array():
    A = np.load('response.npy')
    return A

def load_options_list():
    options = pickle.load(open('options.pickle', 'rb'))
    return options

def random_codes(colors, places, nbr):
    return choices(list(product(ascii_uppercase[:colors], repeat=places)), k=nbr)

def experiment(nbr, all=False):
    import pandas as pd
    colors = 6
    places = 4
    codes, rounds, times = [], [], []
    runs = 0
    if all == False:
        to_play = random_codes(colors, places, nbr)
    elif all == True:
        to_play = list(product(ascii_uppercase[:colors], repeat=places))
    for code in to_play:
        start_time = time.time()
        game = Game(code, colors)
        game.play()
        stop_time = time.time()
        codes.append(code)
        rounds.append(game.round)
        times.append(stop_time-start_time)
        runs += 1

    df = pd.DataFrame({'code': codes,
                       'rounds': rounds,
                       'time': times})
    print(f'Runs: {runs}, avg tries: {df.rounds.mean()}')
    print(df.rounds.describe())
    # df.to_csv(f'results_Knuth_experiment_{runs}.csv', sep=';')
    return df

if __name__ == '__main__':
    # experiment(1000)
    # experiment(1000, all=True)

    # game = Game('FDFA', 6)
    # game.play()

    game = Game('AAAC', 6)
    game.play()
