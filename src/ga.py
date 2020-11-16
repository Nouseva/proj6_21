import copy
import heapq
import metrics
import multiprocessing.pool as mpool
import os
import random
import shutil
import time
import math

width = 200
flag_col = width - 1
height = 16

options = [
    "-",  # an empty space
    "X",  # a solid wall
    "?",  # a question mark block with a coin
    "M",  # a question mark block with a mushroom
    "B",  # a breakable block
    "o",  # a coin
    "|",  # a pipe segment
    "T",  # a pipe top
    "E",  # an enemy
    #"f",  # a flag, do not generate
    #"v",  # a flagpole, do not generate
    #"m"  # mario's start position, do not generate
]

single_options = [
    "-",  # an empty space
    "X",  # a solid wall
    "?",  # a question mark block with a coin
    "M",  # a question mark block with a mushroom
    "B",  # a breakable block
    "o",  # a coin
    #"|",  # a pipe segment
    #"T",  # a pipe top
    "E",  # an enemy
    #"f",  # a flag, do not generate
    #"v",  # a flagpole, do not generate
    #"m"  # mario's start position, do not generate
]

class Individual_Grid(object):
    __slots__ = ["genome", "_fitness"]

    def __init__(self, genome):
        self.genome = copy.deepcopy(genome)
        self._fitness = None

    # Update this individual's estimate of its fitness.
    # This can be expensive so we do it once and then cache the result.
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Print out the possible measurements or look at the implementation of metrics.py for other keys:
        # print(measurements.keys())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Modify this, and possibly add more metrics.  You can replace this with whatever code you like.
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients))
        return self

    # Return the cached fitness value or calculate it as needed.
    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # Mutate a genome into a new genome.  Note that this is a _genome_, not an individual!
    def mutate(self, new_genome):
        # How likely a genome is to be modified
        mutation_chance = 0.4
        # How much of a genome is modified if it is
        mutation_percentage = 0.3
        # STUDENT implement a mutation operator, also consider not mutating this individual
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc

        if random.random() < mutation_chance:
            to_change = math.floor(mutation_percentage * width)

            # TODO: Rewrite to fit with Grid representation
            #       - For each row, try and change to_change amount of entries
            #       - Check if targeted entry would result in inconsistencies: floating pipes, no finish, etc...
            #       - have a mutation preference for each entry to be changed?: mushroom -> coin, air -> enemy, etc...
            for row, gene_strips in enumerate(new_genome):
                # Leave the base alone
                if row == height - 1:
                    break
                # Select the elements in the row that will be changed
                to_mutate = random.sample(list(enumerate(gene_strips)), to_change)

                for col, gene in to_mutate:
                    # Dont change anything from the flag onward
                    if col >= flag_col - 1:
                        continue

                    # Pipe Top will become air, and move down one space if available
                    if gene == 'T':
                        new_genome[row][col] = '-'
                        if new_genome[row+1][col] == '|':
                            new_genome[row+1][col] = 'T'

                    # Pipe mid will either become the new top of a pipe, or change into the new base
                    elif gene == '|':
                        if random.random() < .5:
                            new_genome[row][col] = 'T'
                            for r in reversed(range(row)):
                                prev_gene = new_genome[r][col]
                                new_genome[r][col] = random.choice(single_options)
                                if prev_gene == 'T':
                                    break

                        else:
                            new_genome[row][col] = 'X'
                            new_genome[row][col+1] = 'X'

                            # for all genes until prev pipe base, change randomly
                            for r in range(row+1, height):
                                if new_genome[r][col] == 'X':
                                    break
                                new_genome[r][col] = random.choice(single_options)
                                new_genome[r][col+1] = random.choice(single_options)


                    elif gene == '-':
                        if col > 0:
                            # dont change the empty space next to a pipe
                            if new_genome[row][col-1] == 'T' or new_genome[row][col-1] == '|':
                                continue
                        new_genome[row][col] = random.choice(single_options)

                    else:
                        new_genome[row][col] = '-'

        return new_genome

    # Create zero or more children from self and other
    def generate_children(self, other):
        # Leaving first and last columns alone...
        # do crossover with other
        #print('new genome')
        #print(new_genome)
        # print(other)
        new_gen1 = []
        new_gen2 = []
        left = 1
        right = width - 1
        # print(len(new_genome), width)

        # Single-point crossover

        # Select a random crossover point
        cross_point = math.floor(random.random() * right)

        # For each row, slice the lists and append them to the new genomes
        for y in range(height):
            new_gen1.append(self.genome[y][:cross_point] + other.genome[y][cross_point:])
            new_gen2.append(other.genome[y][:cross_point] + self.genome[y][cross_point:])

        """
        for y in range(height):
            for x in range(left, right):
                # STUDENT Which one should you take?  Self, or other?  Why?
                # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
                selected = random.choice([self.genome, other.genome])
                # print(selected)
                new_genome[y][x] = selected[y][x]

                pass
        """
        # Return all the possible children formed with the parent genomes
        return ( Individual_Grid(self.mutate(new_gen1)), Individual_Grid(self.mutate(new_gen2)) )

    # Turn the genome into a level string (easy for this genome)
    def to_level(self):
        return self.genome

    # These both start with every floor tile filled with Xs
    # STUDENT Feel free to change these
    @classmethod
    def empty_individual(cls):
        g = [["-" for col in range(width)] for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][flag_col] = "v"
        for col in range(8, 14):
            g[col][flag_col] = "f"
        for col in range(14, 16):
            g[col][flag_col] = "X"
        return cls(g)

    @classmethod
    def random_individual(cls):
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        g = [random.choices(options, k=width) for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][flag_col] = "v"
        for row in range(8, 14):
            g[row][flag_col] = "f"
        for row in range(14, 16):
            g[row][flag_col] = "X"

        # Clean up to ensure no floating pipes
        for row in range(height-1):
            for col in range(flag_col):
                # Spaces have no requirements
                if g[row][col] == '-':
                    continue

                # Special cases when in column that preceeds the flag
                if col + 1 == flag_col:
                    # Pipes are not allowed in the column preceeding the flag
                    if g[row][col] == 'T' or g[row][col] == '|':
                        g[row][col] = '-'

                # Cases where right neighbor can be modified
                else:
                    # Pipe must have a tube or solid base below it, empty space next to it
                    if g[row][col] == 'T' or g[row][col] == '|':
                        # If there is no base, add pipe down
                        if not (g[row+1][col] == 'X' or g[row+1][col] == '|'):
                            g[row+1][col] = '|'
                        else:
                            # Current pipe piece has a base, ensure that base is two wide
                            # if base is actually another pipe piece, then it will be cleaned by that piece
                            g[row+1][col+1] = 'X'
                        g[row][col+1] = '-'


        # g[8:14][-1] = ["f"] * 6
        # g[14:16][-1] = ["X", "X"]
        return cls(g)


def offset_by_upto(val, variance, min=None, max=None):
    val += random.normalvariate(0, variance**0.5)
    if min is not None and val < min:
        val = min
    if max is not None and val > max:
        val = max
    return int(val)


def clip(lo, val, hi):
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

# Inspired by https://www.researchgate.net/profile/Philippe_Pasquier/publication/220867545_Towards_a_Generic_Framework_for_Automated_Video_Game_Level_Creation/links/0912f510ac2bed57d1000000.pdf


class Individual_DE(object):
    # Calculating the level isn't cheap either so we cache it too.
    __slots__ = ["genome", "_fitness", "_level"]

    # Genome is a heapq of design elements sorted by X, then type, then other parameters
    def __init__(self, genome):
        self.genome = list(genome)
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None

    # Calculate and cache fitness
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Add more metrics?
        # STUDENT Improve this with any code you like
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        penalties = 0
        # STUDENT For example, too many stairs are unaesthetic.  Let's penalize that
        if len(list(filter(lambda de: de[1] == "6_stairs", self.genome))) > 5:
            penalties -= 2
        # STUDENT If you go for the FI-2POP extra credit, you can put constraint calculation in here too and cache it in a new entry in __slots__.
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients)) + penalties
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, new_genome):
        # STUDENT How does this work?  Explain it in your writeup.
        # STUDENT consider putting more constraints on this, to prevent generating weird things
        if random.random() < 0.1 and len(new_genome) > 0:
            # select one "gene" of the genome at random
            to_change = random.randint(0, len(new_genome) - 1)
            de = new_genome[to_change]
            # make a new "gene" by copying the "gene" from the original genome for later "allele" modification
            new_de = de
            # get x coord from original "gene"
            x = de[0]
            # get design element type from original "gene"
            de_type = de[1]
            # get random float [0.0, 1.0)
            choice = random.random()
            if de_type == "4_block":
                # for a block type DE, get y coord, breakability aspects from original gene
                y = de[2]
                breakable = de[3]
                # 1/3 probability to offset x coord 
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                # 1/3 probability to offset y coord
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                # 1/3 probability to reverse breakability
                else:
                    breakable = not de[3]
                # place resulting "allele" in the new "gene"
                new_de = (x, de_type, y, breakable)
            elif de_type == "5_qblock":
                # for a ? block type DE, get y coord, powerup aspects from original gene
                y = de[2]
                has_powerup = de[3]  # boolean
                # 1/3 probability to offset x coord
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                # 1/3 probability to offset y coord
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                # 1/3 probability to reverse powerup presence
                else:
                    has_powerup = not de[3]
                # place resulting "allele" in the new "gene"
                new_de = (x, de_type, y, has_powerup)
            elif de_type == "3_coin":
                # for a coin type DE, get y coord
                y = de[2]
                # 1/2 probability to offset x coord
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                # 1/2 probability to offset y coord
                else:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                new_de = (x, de_type, y)
            elif de_type == "7_pipe":
                # for a pipe type DE, get height aspect
                h = de[2]
                # 1/2 probability to offset x coord
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                # 1/2 probability to offset height aspect
                else:
                    h = offset_by_upto(h, 2, min=2, max=height - 4)
                # place resulting "allele" in the new "gene"
                new_de = (x, de_type, h)
            elif de_type == "0_hole":
                # for a hole type DE, get width aspect
                w = de[2]
                # 1/2 probability to offset x coord
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                # 1/2 probability to offset width aspect
                else:
                    w = offset_by_upto(w, 4, min=1, max=width - 2)
                # place resulting "allele" in the new "gene"
                new_de = (x, de_type, w)
            elif de_type == "6_stairs":
                # for a stair type DE, get height, direction aspects
                h = de[2]
                dx = de[3]  # -1 or 1
                # 1/3 probability to offset x coord
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                # 1/3 probability to offset height aspect
                elif choice < 0.66:
                    h = offset_by_upto(h, 8, min=1, max=height - 4)
                # 1/3 probability to reverse direction
                else:
                    dx = -dx
                # place resulting "allele" in the new "gene"
                new_de = (x, de_type, h, dx)
            elif de_type == "1_platform":
                #for a platform type DE, get width, y coord, and block type aspects
                w = de[2]
                y = de[3]
                madeof = de[4]  # from "?", "X", "B"
                # 1/4 probability to offset x coord
                if choice < 0.25:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                # 1/4 probability to offset width aspect
                elif choice < 0.5:
                    w = offset_by_upto(w, 8, min=1, max=width - 2)
                # 1/4 probability to offset y coord
                elif choice < 0.75:
                    y = offset_by_upto(y, height, min=0, max=height - 1)
                # 1/4 probability to reselect block type at random
                else:
                    madeof = random.choice(["?", "X", "B"])
                # place resulting "allele" in the new "gene"
                new_de = (x, de_type, w, y, madeof)
            elif de_type == "2_enemy":
                # for an enemy type DE, do nothing
                pass
            # pop the original gene from the new genome, push the mutated new gene onto the new genome, then return the new genome
            new_genome.pop(to_change)
            heapq.heappush(new_genome, new_de)
        return new_genome

    def generate_children(self, other):
        # STUDENT How does this work?  Explain it in your writeup.
        # pick a random crossover point in each parent genome at random
        pa = random.randint(0, len(self.genome) - 1)
        pb = random.randint(0, len(other.genome) - 1)
        # take the portion of parent a before crossover and the portion of
        # parent b after crossover, combine for generated child a
        a_part = self.genome[:pa] if len(self.genome) > 0 else []
        b_part = other.genome[pb:] if len(other.genome) > 0 else []
        ga = a_part + b_part
        # take the portion of parent b before crossover and the portion of
        # parent a after crossover, combine for generated child b
        b_part = other.genome[:pb] if len(other.genome) > 0 else []
        a_part = self.genome[pa:] if len(self.genome) > 0 else []
        gb = b_part + a_part
        # do mutation
        return Individual_DE(self.mutate(ga)), Individual_DE(self.mutate(gb))

    # Apply the DEs to a base level.
    def to_level(self):
        if self._level is None:
            base = Individual_Grid.empty_individual().to_level()
            for de in sorted(self.genome, key=lambda de: (de[1], de[0], de)):
                # de: x, type, ...
                x = de[0]
                de_type = de[1]
                if de_type == "4_block":
                    y = de[2]
                    breakable = de[3]
                    base[y][x] = "B" if breakable else "X"
                elif de_type == "5_qblock":
                    y = de[2]
                    has_powerup = de[3]  # boolean
                    base[y][x] = "M" if has_powerup else "?"
                elif de_type == "3_coin":
                    y = de[2]
                    base[y][x] = "o"
                elif de_type == "7_pipe":
                    h = de[2]
                    base[height - h - 1][x] = "T"
                    for y in range(height - h, height):
                        base[y][x] = "|"
                elif de_type == "0_hole":
                    w = de[2]
                    for x2 in range(w):
                        base[height - 1][clip(1, x + x2, width - 2)] = "-"
                elif de_type == "6_stairs":
                    h = de[2]
                    dx = de[3]  # -1 or 1
                    for x2 in range(1, h + 1):
                        for y in range(x2 if dx == 1 else h - x2):
                            base[clip(0, height - y - 1, height - 1)][clip(1, x + x2, width - 2)] = "X"
                elif de_type == "1_platform":
                    w = de[2]
                    h = de[3]
                    madeof = de[4]  # from "?", "X", "B"
                    for x2 in range(w):
                        base[clip(0, height - h - 1, height - 1)][clip(1, x + x2, width - 2)] = madeof
                elif de_type == "2_enemy":
                    base[height - 2][x] = "E"
            self._level = base
        return self._level

    @classmethod
    def empty_individual(_cls):
        # STUDENT Maybe enhance this
        g = []
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        # STUDENT Maybe enhance this
        elt_count = random.randint(8, 128)
        g = [random.choice([
            (random.randint(1, width - 2), "0_hole", random.randint(1, 8)),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(0, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "2_enemy"),
            (random.randint(1, width - 2), "3_coin", random.randint(0, height - 1)),
            (random.randint(1, width - 2), "4_block", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "5_qblock", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "6_stairs", random.randint(1, height - 4), random.choice([-1, 1])),
            (random.randint(1, width - 2), "7_pipe", random.randint(2, height - 4))
        ]) for i in range(elt_count)]
        return Individual_DE(g)

# Selector
Individual = Individual_Grid


def generate_successors(population):
    # results = []
    # STUDENT Design and implement this
    # Hint: Call generate_children() on some individuals and fill up results.
    # TODO: some smart method of selecting partners
    # partners are the individuals in the opposing position in the list
    # partners  = population[::-1]

    # partners are next best individual, worst gets to pair with best
    partners  = population[1:] + population[:1]

    groupings = list(zip(population, partners))

    def select_best_child(children):
        children = list(children)
        return max(children, key=lambda c: c.fitness())

    results = list(map(select_best_child, [ p.generate_children(q) for p, q in groupings ]))
    return results


def ga():
    # STUDENT Feel free to play with this parameter
    pop_limit = 480
    # Code to parallelize some computations
    batches = os.cpu_count()
    if pop_limit % batches != 0:
        print("It's ideal if pop_limit divides evenly into " + str(batches) + " batches.")
    batch_size = int(math.ceil(pop_limit / batches))
    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        # STUDENT (Optional) change population initialization
        population = [Individual.random_individual() if random.random() < 0.9
                      else Individual.empty_individual()
                      for _g in range(pop_limit)]
        # But leave this line alone; we have to reassign to population because we get a new population that has more cached stuff in it.
        population = pool.map(Individual.calculate_fitness,
                              population,
                              batch_size)
        init_done = time.time()
        print("Created and calculated initial population statistics in:", init_done - init_time, "seconds")
        generation = 0
        start = time.time()
        now = start
        print("Use ctrl-c to terminate this loop manually.")
        # print(population)
        try:
            while True:
                now = time.time()
                # Print out statistics
                if generation > 0:
                    best = max(population, key=Individual.fitness)
                    print("Generation:", str(generation))
                    print("Max fitness:", str(best.fitness()))
                    print("Average generation time:", (now - start) / generation)
                    print("Net time:", now - start)
                    with open("levels/last.txt", 'w') as f:
                        for row in best.to_level():
                            f.write("".join(row) + "\n")
                generation += 1
                # STUDENT Determine stopping condition
                # TODO: Allow to run some generation
                # stop_condition = False
                if generation > 20:
                    break
                # STUDENT Also consider using FI-2POP as in the Sorenson & Pasquier paper
                gentime = time.time()
                next_population = generate_successors(population)
                gendone = time.time()
                print("Generated successors in:", gendone - gentime, "seconds")
                # Calculate fitness in batches in parallel
                next_population = pool.map(Individual.calculate_fitness,
                                           next_population,
                                           batch_size)
                popdone = time.time()
                print("Calculated fitnesses in:", popdone - gendone, "seconds")
                population = next_population
        except KeyboardInterrupt:
            pass
    return population


if __name__ == "__main__":
    final_gen = sorted(ga(), key=Individual.fitness, reverse=True)
    best = final_gen[0]
    print("Best fitness: " + str(best.fitness()))
    now = time.strftime("%m_%d_%H_%M_%S")
    # STUDENT You can change this if you want to blast out the whole generation, or ten random samples, or...
    for k in range(0, 1):
        with open("levels/" + now + "_" + str(k) + ".txt", 'w') as f:
            for row in final_gen[k].to_level():
                f.write("".join(row) + "\n")
