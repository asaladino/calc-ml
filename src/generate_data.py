import pandas as pd
import random


class GenerateData:
    def __init__(self, lower_bound=0, upper_bound=1000, results=10000, file_name='training_data.csv'):
        self.outputFolder = '../data/'
        self.fileName = file_name

        # 0 = addition
        # 1 = subtraction
        # 2 = multiplication
        # 3 = division
        self.operations = [0]

        self.index = []
        self.number_1 = []
        self.operator = []
        self.number_2 = []
        self.solution = []

        self.lowerBound = lower_bound
        self.upperBound = upper_bound
        # Number of generated results
        self.results = results

    def build(self):
        self.index = range(self.results)
        self.number_1 = []
        self.operator = []
        self.number_2 = []
        self.solution = []
        for _ in self.index:
            number_1 = random.randint(self.lowerBound, self.upperBound)
            number_2 = random.randint(self.lowerBound, self.upperBound)
            operator = random.randint(0, len(self.operations) - 1)
            solution = self.calc(number_1, operator, number_2)

            self.number_1.append(number_1)
            self.operator.append(operator)
            self.number_2.append(number_2)
            self.solution.append(solution)

    def save(self):
        data = {
            'number_1': self.number_1,
            'operator': self.operator,
            'number_2': self.number_2,
            'solution': self.solution
        }
        df = pd.DataFrame(data, index=self.index)
        df.drop_duplicates()
        df.index.name = 'id'
        print('Saving to {}{}'.format(self.outputFolder, self.fileName))
        print(df.head())
        df.to_csv(self.outputFolder + self.fileName)

    @staticmethod
    def calc(number_1, operator, number_2):
        if operator is 0:
            return number_1 + number_2
        if operator is 1:
            return number_1 - number_2
        if operator is 2:
            return number_1 * number_2
        if operator is 3:
            return number_1 / number_2
        raise ValueError('Unknown operation: {}'.format(operator))


generate_data = GenerateData()
generate_data.build()
generate_data.save()

generate_data.fileName = 'test_data.csv'
generate_data.lowerBound = generate_data.upperBound
generate_data.upperBound = 2 * generate_data.upperBound
generate_data.build()
generate_data.save()
