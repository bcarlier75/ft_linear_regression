from sys import argv
from maths import *
import matplotlib.pyplot as plt
import csv


def print_usage():
    print(f'Usage: python train.py [-h] [-p] [-m]')
    print(f'\t -h : display this message')
    print(f'\t -p : display plot')
    print(f'\t -m : display model metrics')


def check_args():
    flag_plot = 0
    flag_metrics = 0
    for i in range(1, len(argv)):
        if argv[i] == '-h':
            print_usage()
            return -1, -1
        elif argv[i] == '-p':
            flag_plot = 1
        elif argv[i] == '-m':
            flag_metrics = 1
    return flag_plot, flag_metrics


def get_data():
    mileage = []
    price = []
    with open('csv/data.csv', 'r') as f:
        for index, line in enumerate(f):
            if index != 0:
                tab = line.rstrip('\n').split(',')
                try:
                    mileage.append(float(tab[0]))
                    price.append(float(tab[1]))
                except ValueError:
                    print(f'Dataset wrongly formated !')
                    return -1, -1
    return mileage, price


def get_thetas():
    tab = [0, 0]
    with open('csv/thetas_file.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            row[0] = float(row[0])
            row[1] = float(row[1])
            return row
        return tab


def write_thetas(theta_0: float, theta_1: float):
    with open('csv/thetas_file.csv', 'w') as f:
        f.write(str(theta_0) + ',' + str(theta_1))


def display_plot(mileage: List[float], price: List[float], mse: List[float]):
    plt.plot(mileage, mse, c='aquamarine')
    plt.scatter(mileage, price, s=10, c='navy')
    plt.xlabel('mileage')
    plt.ylabel('price')
    plt.show()


def display_metrics(theta_0: float, theta_1: float, mileage: List[float], price: List[float]):
    rsq = r_squared(theta_0, theta_1, mileage, price)
    print(f'Coefficient : {theta_1:.8f}')
    print(f'Intercept   : {theta_0:.8f}')
    print(f'R_squared   : {rsq}')
