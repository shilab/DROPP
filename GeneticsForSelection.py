# File: genetic.py
#    from chapter 12 of _Genetic Algorithms with Python_
#
# Author: Clinton Sheppard <fluentcoder@gmail.com>
# Copyright (c) 2016 Clinton Sheppard
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.

import random
import statistics
import sys
import time
from bisect import bisect_left
from enum import Enum
from math import exp
from mUtils import Queue

class Genetics:

    def __init__(self, get_fitness):
        self.MiniTimer = 0
        self.Parents = []
        self.HistoricalFitnesses = []
        self.FnGetFitness = get_fitness
        self.BestParent = None
        self.IdleCounter = 0


    def __recalculate_fitnesses(self):
        for parent in self.Parents:
            parent.Fitness = self.FnGetFitness(parent.Genes)
        self.BestParent.Fitness = self.FnGetFitness(self.BestParent.Genes)
        pass

    def reset_mini_timer(self, recalculateFitnesses=True):
        self.MiniTimer = time.time()
        if recalculateFitnesses:
            self.__recalculate_fitnesses()

    def _mutate_custom(self, parent, custom_mutate):
        # childGenes = np.copy(parent.Genes)
        childGenes, fitness = custom_mutate(parent.Genes, fitness=parent.Fitness)
        # fitness = get_fitness(childGenes)
        return Chromosome(childGenes, fitness, Strategies.Mutate)


    def _crossover(self, parent, index, get_fitness, crossover, mutate,
                   generate_parent):
        donorIndex = random.randrange(0, len(self.Parents))
        if donorIndex == index:
            donorIndex = (donorIndex + 1) % len(self.Parents)

        childGenes, fitness = crossover(parent.Genes, self.Parents[donorIndex].Genes, parent.Fitness,
                                        self.Parents[donorIndex].Fitness)
        if childGenes is None:
            # parent and donor are indistinguishable
            self.Parents[donorIndex] = generate_parent(donorIndex)
            return mutate(self.Parents[index])
        # fitness = get_fitness(childGenes)
        return Chromosome(childGenes, fitness, Strategies.Crossover)


    def get_best(self, targetLen, optimalFitness, display, triggerLimit=None,
                 custom_mutate=None, custom_create=None, maxAge=None,
                 poolSize=1, crossover=None, maxSeconds=None, maxIdleRounds=None,
                 maxSecondsMiniStroke=None,
                 staticProbabilities=False,
                 probs=None):
        def fnMutate(parent):
            return self._mutate_custom(parent, custom_mutate)

        def fnGenerateParent(index):
            genes = custom_create(index)
            return Chromosome(genes, self.FnGetFitness(genes), Strategies.Create)

        strategyLookup = {
            Strategies.Create: lambda p, i: fnGenerateParent(i),
            Strategies.Mutate: lambda p, i: fnMutate(p),
            Strategies.Crossover: lambda p, i:
            self._crossover(p, i, self.FnGetFitness, crossover, fnMutate,
                            fnGenerateParent)
        }
        usedStrategies = Queue(5)
        usedStrategies.enqueue(strategyLookup[Strategies.Mutate])
        if crossover is not None:
            usedStrategies.enqueue(strategyLookup[Strategies.Crossover])

            def fnNewChild(parent, index):
                return random.choice(usedStrategies)(parent, index)
        else:
            def fnNewChild(parent, index):
                return fnMutate(parent)
        usedStrategies.lock_bound()

        for timedOut, improvement, pindex in self._get_improvement(fnNewChild, fnGenerateParent,
                                                              maxAge, poolSize, maxSeconds, maxIdleRounds,
                                                                   maxSecondsMiniStroke):
            if timedOut == 1: #real timeout
                yield timedOut, improvement, improvement.Fitness.Metric, improvement.Fitness.NOG
                break
            elif timedOut == 2 or timedOut == 3:#mini/idle timeout
                yield timedOut, improvement, improvement.Fitness.Metric, improvement.Fitness.NOG
                break
            else:
                display(improvement, pindex)
                if improvement.Strategy != Strategies.Create:
                    f = strategyLookup[improvement.Strategy]
                    usedStrategies.enqueue(f)
                yield timedOut, improvement, improvement.Fitness.Metric, improvement.Fitness.NOG
                if not optimalFitness > improvement.Fitness:
                    break


    def _get_improvement(self, new_child, generate_parent, maxAge, poolSize, maxSeconds, maxIdleRounds,
                         maxSecondsMiniStroke):
        startTime = time.time()

        self.MiniTimer = time.time()
        self.IdleCounter = 0
        self.BestParent = generate_parent(0)
        self.Parents.append(self.BestParent)
        yield 0, self.BestParent, 0
        self.HistoricalFitnesses.append(self.BestParent.Fitness)
        for pi in range(poolSize - 1):
            parent = generate_parent(pi+1)
            if maxSeconds is not None and time.time() - startTime > maxSeconds:
                yield 1, parent, 0#pi + 1
            if  maxSecondsMiniStroke is not None and time.time() - self.MiniTimer > maxSecondsMiniStroke:
                yield 2, parent, 0#pi + 1
            if maxIdleRounds is not None and self.IdleCounter > maxIdleRounds:
                # self.IdleCounter = 0
                yield 3, self.BestParent, 0#pi + 1
                self.__recalculate_fitnesses()
            if parent.Fitness > self.BestParent.Fitness:
                yield 0, parent, 0#pi + 1
                self.BestParent = parent
                self.HistoricalFitnesses.append(parent.Fitness)
            self.Parents.append(parent)
        lastParentIndex = poolSize - 1
        pindex = 1
        while True:
            self.IdleCounter += 1
            if maxSeconds is not None and time.time() - startTime > maxSeconds:
                yield 1, self.BestParent, self.IdleCounter//poolSize + 1#pindex
            if maxSecondsMiniStroke is not None and time.time() - self.MiniTimer > maxSecondsMiniStroke:
                yield 2, self.BestParent, self.IdleCounter//poolSize + 1#pindex
            if maxIdleRounds is not None and self.IdleCounter > maxIdleRounds:
                # self.IdleCounter = 0
                yield 3, self.BestParent, self.IdleCounter//poolSize + 1#pindex
                self.__recalculate_fitnesses()
            pindex = pindex - 1 if pindex > 0 else lastParentIndex
            parent = self.Parents[pindex]
            child = new_child(parent, pindex)
            if parent.Fitness > child.Fitness:
                if maxAge is None:
                    continue
                parent.Age += 1
                if maxAge > parent.Age:
                    continue
                index = bisect_left(self.HistoricalFitnesses, child.Fitness, 0,
                                    len(self.HistoricalFitnesses))
                proportionSimilar = index / len(self.HistoricalFitnesses)
                if random.random() < exp(-proportionSimilar):
                    self.Parents[pindex] = child
                    # print("Parent discarded at:", pindex)
                    continue
                self.BestParent.Age = 0
                self.Parents[pindex] = self.BestParent
                continue
            if not child.Fitness > parent.Fitness:
                # same fitness
                child.Age = parent.Age + 1
                self.Parents[pindex] = child
                continue
            child.Age = 0
            self.Parents[pindex] = child

            if child.Fitness > self.BestParent.Fitness:
                # self.IdleCounter = 0
                self.BestParent = child
                yield 0, self.BestParent, self.IdleCounter//poolSize + 1#pindex
                self.HistoricalFitnesses.append(self.BestParent.Fitness)


class Chromosome:
    def __init__(self, genes, fitness, strategy):
        self.Genes = genes
        self.Fitness = fitness
        self.Strategy = strategy
        self.Age = 0


class Strategies(Enum):
    Create = 0,
    Mutate = 1,
    Crossover = 2


class Benchmark:
    @staticmethod
    def run(function):
        timings = []
        stdout = sys.stdout
        for i in range(100):
            sys.stdout = None
            startTime = time.time()
            function()
            seconds = time.time() - startTime
            sys.stdout = stdout
            timings.append(seconds)
            mean = statistics.mean(timings)
            if i < 10 or i % 10 == 9:
                print("{} {:3.2f} {:3.2f}".format(
                    1 + i, mean,
                    statistics.stdev(timings, mean) if i > 1 else 0))
