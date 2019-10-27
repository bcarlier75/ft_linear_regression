from utils import check_args, get_data, write_thetas, display_metrics, display_plot
from maths import List, error, predict
from typing import Tuple


def gradient_step(theta_0: float, theta_1: float,
                  grad_t_0: float, grad_t_1: float,
                  learning_rate: float,
                  m: int) -> Tuple[float, float]:
    tmp_theta_0 = theta_0 - (learning_rate * ((1.0 / m) * grad_t_0))
    tmp_theta_1 = theta_1 - (learning_rate * ((1.0 / m) * grad_t_1))
    return tmp_theta_0, tmp_theta_1


def gradient_descent(mileage: List[float], price: List[float],
                     l_r: float) -> Tuple[float, float]:
    m = len(mileage)
    theta_0 = 0.0
    theta_1 = 0.0
    grad_t_1 = 0.0
    while grad_t_1 != sum(error(theta_0, theta_1, x_i, y_i) * x_i
                          for x_i, y_i in list(zip(mileage, price))):
        # Partial derivative of loss with respect to theta_0
        grad_t_0 = sum(error(theta_0, theta_1, x_i, y_i)
                       for x_i, y_i in list(zip(mileage, price)))
        # Partial derivative of loss with respect to theta_1
        grad_t_1 = sum(error(theta_0, theta_1, x_i, y_i) * x_i
                       for x_i, y_i in list(zip(mileage, price)))
        # Update of theta_0 and theta_1 by performing a gradien step
        theta_0, theta_1 = gradient_step(theta_0, theta_1, grad_t_0, grad_t_1, l_r, m)

    return theta_0, theta_1


def main():
    flag_plot, flag_metrics = check_args()
    if (flag_plot, flag_metrics) == (-1, -1):
        return
    (mileage, price) = get_data()
    if (mileage, price) == (-1, -1):
        return
    # Normalize mileage and price for faster computation
    mileage_norm = [float(mileage[i])/max(mileage) for i in range(len(mileage))]
    price_norm = [float(price[i])/max(price) for i in range(len(price))]
    # Perform gradient descent. Learning rate can be tuned here.
    [theta_0, theta_1] = gradient_descent(mileage_norm, price_norm, l_r=0.1)
    # Denormalize
    theta_0 = theta_0 * max(price)
    theta_1 = theta_1 * (max(price) / max(mileage))
    # Input theta values in the .csv file
    write_thetas(theta_0, theta_1)

    # Display information according to flags
    if flag_metrics == 1:
        display_metrics(theta_0, theta_1, mileage, price)
    if flag_plot == 1:
        mse = list()
        for i in range(len(mileage)):
            mse.append(predict(theta_0, theta_1, mileage[i]))
        display_plot(mileage, price, mse)


if __name__ == "__main__":
    main()
