import random
import math
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    n = 500

    # Generate 500 values between 1 and 100
    data = [random.randint(1, 100) for _ in range(n)]

    full_mean = sum(data) / n

    temp = [(x - full_mean) ** 2 for x in data]
    full_stand_des = math.sqrt(sum(temp) / n)

    known_mean = 0
    known_variation = 0
    k = 0

    mean_trajectory = []
    stand_des_trajectory = []

    for i, x in enumerate(data):
        mean_k_next = (k * known_mean + data[i + 1]) / (k + 1)
        variation_k_next = ((k * known_variation) + ((data[i + 1] - mean_k_next) ** 2)) / (k + 1)

        mean_trajectory.append(mean_k_next)
        stand_des_trajectory.append(math.sqrt(variation_k_next))

        known_mean = mean_k_next
        known_variation = variation_k_next
        k += 1

        if i == (len(data) - 2):
            break

    print(full_mean)
    print(full_stand_des)
    print(mean_trajectory[len(mean_trajectory) - 1])
    print(stand_des_trajectory[len(stand_des_trajectory) - 1])

    x_values = np.linspace(0, n, n)

    y_mean_full = np.full_like(x_values, full_mean)
    plt.plot(x_values, y_mean_full, 'r', label='full dataset')
    plt.plot(x_values[:-1], mean_trajectory, 'b', label='dynamic dataset')

    # add tags and title
    plt.xlabel('Values from 0 to N')
    plt.ylabel('Mean values')
    plt.title('Graph to compare Means')

    # Add legend
    plt.legend()
    # Show graph
    plt.show()

    y_full_stand_des = np.full_like(x_values, full_stand_des)
    plt.plot(x_values, y_full_stand_des, 'r', label='full dataset')
    plt.plot(x_values[:-1], stand_des_trajectory, 'b', label='dynamic dataset')

    # add tags and title
    plt.xlabel('Values from 0 to N')
    plt.ylabel('Standard Deviation values')
    plt.title('Graph to compare Standard Deviations')

    # Add legend
    plt.legend()
    # Show graph
    plt.show()
    pass
