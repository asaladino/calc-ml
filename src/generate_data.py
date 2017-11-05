import pandas as pd

class GenerateData:

    def __init__(self):
        self.outputFolder = '../data/'

        # 0 = addition
        # 1 = subtraction
        # 2 = multiplication
        # 3 = division
        self.operations = [0]
        self.header = ['number_1', 'operation', 'number_2', 'solution']
        self.data = []

        self.lowerBound = 0
        self.upperBound = 1000
        # Number of generated results
        self.results = 10000

    def build(self):
        self.data = [self.header]
        for index in self.results:

            self.data.append()