#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
moving_average = __import__('4-moving_average').moving_average


if __name__ == '__main__':
    # Sample temperature data
    data = [72, 78, 71, 68, 66, 69, 79, 79, 65, 64, 66, 78, 64, 64,
            81, 71, 69, 65, 72, 64, 60, 61, 62, 66, 72, 72, 67, 67,
            67, 68, 75]

    days = list(range(1, len(data) + 1))

    # Compute moving average
    m_avg = moving_average(data, 0.9)

    # Print results
    print(m_avg)

    # Plot actual data vs. moving average
    plt.plot(days, data, 'r', days, m_avg, 'b')
    plt.xlabel('Day of Month')
    plt.ylabel('Temperature (Fahrenheit)')
    plt.title('SF Maximum Temperatures in October 2018')
    plt.legend(['Actual', 'Moving Average'])
    plt.show()
