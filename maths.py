from typing import List

Vector = List[float]


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def de_mean(xs: List[float]) -> List[float]:
    x_bar = mean(xs)
    return [x - x_bar for x in xs]


def total_sum_of_squares(y: Vector) -> float:
    return sum(v ** 2 for v in de_mean(y))


def predict(theta_0: float, theta_1: float, x_i: float) -> float:
    return theta_1 * x_i + theta_0


def error(theta_0: float, theta_1: float, x_i: float, y_i: float) -> float:
    return predict(theta_0, theta_1, x_i) - y_i


def sum_of_sqerrors(theta_0: float, theta_1: float, x: Vector, y: Vector) -> float:
    return sum(error(theta_0, theta_1, x_i, y_i) ** 2
               for x_i, y_i in list(zip(x, y)))


def r_squared(theta_0: float, theta_1: float, x: Vector, y: Vector) -> float:
    return 1.0 - (sum_of_sqerrors(theta_0, theta_1, x, y) /
                  total_sum_of_squares(y))
