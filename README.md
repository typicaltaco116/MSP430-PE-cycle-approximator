# NOR Flash Cycle Estimator Python Scripts

## cluster-approximator.py
Python script intended to approximate the number of program erase cycles a particular MSP430F5529 NOR Flash segment as experienced. Takes multiple input CSV files and outputs the approximate PE cycles to stdout and creates a MatPlotLib window reporting the details.

```cluster-approximator.py [options] [file ...]```

Options:
- -m [model]    Sets the regression model that the approximator will utilize. Possible values {linear, exponential}. Program will default to exponential regression if no model is specified.

## gold-cluster-generator.py
Python script that generates a csv file containing the centroid clusters of the input csv files. Output is directed towards stdout and is not configurable currently.

```gold-cluster-generator.py [options] [file ...]```

Options:
- -m [model]    Sets the regression model
