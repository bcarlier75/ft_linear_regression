from utils import get_thetas, predict


def main():
    theta_0, theta_1 = get_thetas()
    mileage = 0
    while mileage <= 0:
        mileage = input(f'Please input a positive mileage: ')
        try:
            mileage = float(mileage)
        except ValueError:
            print(f'You did not input a valid mileage, please try again.\n')
            mileage = 0
        if mileage < 0:
            print(f'You did not input a positive mileage, please try again.\n')

    price = predict(theta_0, theta_1,  mileage)
    if price < 0:
        print(f'\nA car with this mileage is worthless!')
    else:
        print(f'\nEstimated price: {price:.2f} €')


if __name__ == "__main__":
    main()
