# Bayesian Estimation
This repository is for projects solving the practices assigned
at the bayesian estimation class of the Master in Computer Science.

## Practice 1

The objective of this practice is to compare the behavior of
mean and standard deviation when the dataset is known
completely and when the data is arriving dynamically.

__[Trello Board](https://trello.com/b/OLBWVFqw/bayesian-estimation-practice-1)__ - Tasks
organization for this assignment.

When the data set is known from the beginning the procedure is 
quite straight forward. 

```
# Mean when the complete dataset is known
full_mean = sum(data) / n
```

```
# Standard Deviation when the complete dataset is known
temp = [(x - full_mean) ** 2 for x in data]
full_stand_des = math.sqrt(sum(temp) / n)
```

But when the data is arriving as time passes then it is necessary
to use another method to calculate the mean and the standard deviation
with the data known at a certain moment in time.

```
# Formulas that calculates the next mean based on the actual known mean
mean_k_next = (k * known_mean + data[i + 1]) / (k + 1)
```

```
# Formulas that calculates the next variation based on the actual known mean
variation_k_next = ((k * known_variation) + ((data[i + 1] - mean_k_next) ** 2)) / (k + 1)
```

In this case, as new data arrives, the mean and the deviation known so far
are used to calculate the new mean and standard deviation values. All values
found are stored to later be able to graph them and know if the two
methods come to the same solution.

![Mean](Practice%201/Mean.png)

![Deviation](Practice%201/Standard%20Deviation.png)

As can be seen in the images in both cases the result is
practically the same.

## Practice 2

Analytical demonstration that the formulas for calculating the mean when the
complete data set is known is equivalent to the formula for calculating the 
mean when the data arrive dynamically.

__[Trello Board](https://trello.com/b/S8qNeyPh/bayesian-estimation-practice-2)__ - Tasks
organization for this assignment.
