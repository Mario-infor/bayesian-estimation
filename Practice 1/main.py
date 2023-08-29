import random

if __name__ == '__main__':
    # Generate 500 values between 1 and 100
    data = [random.randint(1, 100) for _ in range(10)]

    n = len(data)

    mean = sum(data) / n

    temp = [(x - mean) ** 2 for x in data]
    stand_des = sum(temp) / n
