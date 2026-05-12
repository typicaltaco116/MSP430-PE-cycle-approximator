#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import sys

from csv_processor import getDataCSV
from curve_fitting import getDivisionExpoFit, getDivisionLinearFit

def main():
    if (len(sys.argv) < 2):
        print("Error: No input arguments specified")
        exit()
    model_option = 'linear'

    filenameList = []

    # Get data from all files and add to main list
    option_value_flag = False
    for argc in range(1, len(sys.argv)):
        arg = sys.argv[argc]

        #Process Options if available
        if (option_value_flag):
            if (sys.argv[argc - 1] == '-m'):
                model_option = sys.argv[argc];
            option_value_flag = False

        else:
            #Determine options
            if (arg == '-m'):
                option_value_flag = True

            #No option present
            else:
                filenameList.append(arg)


    divisionSeriesList = []
    timeSeriesList = []

    divisionSizes = [20, 20, 20]
    startIndexes = [1, 23, 45]
    goldReferenceCycles = range(10000, 210000, 10000)

    #Open and gather data from all files
    for file in filenameList:
        divisionData, timeSeries = getDataCSV(file, divisionSizes, startIndexes)
        divisionSeriesList.append(divisionData[0])
        divisionSeriesList.append(divisionData[1])
        divisionSeriesList.append(divisionData[2])
        timeSeriesList.append(timeSeries) # append three times to keep lists the same size
        timeSeriesList.append(timeSeries)
        timeSeriesList.append(timeSeries)


    #Array containing all the points for each stressed segment
    # Each Row contains the ordered pairs for a different cycle count segment
    # points = []
    
    if (model_option == 'linear'):
        # Initial sizing of points list
        m_values, b_values = getDivisionLinearFit(divisionSeriesList[0], timeSeriesList[0])
        points = np.column_stack((m_values, b_values))
        points = points[:, :, np.newaxis]

        #Build rest of list
        for BER_series, time_series in zip(divisionSeriesList[1:], timeSeriesList[1:]):
            m_values, b_values = getDivisionLinearFit(BER_series, time_series)
            mb_doubles = np.column_stack((m_values, b_values))
            mb_doubles = mb_doubles[:, :, np.newaxis]
            points = np.concatenate((points, mb_doubles), axis=2)
    else:
        # Initial sizing of points list
        l_values, a_values = getDivisionExpoFit(divisionSeriesList[0], timeSeriesList[0])
        points = np.column_stack((l_values, a_values))
        points = points[:, :, np.newaxis]

        #Build rest of list
        for BER_series, time_series in zip(divisionSeriesList[1:], timeSeriesList[1:]):
            l_values, a_values = getDivisionLinearFit(BER_series, time_series)
            mb_doubles = np.column_stack((l_values, a_values))
            mb_doubles = mb_doubles[:, :, np.newaxis]
            points = np.concatenate((points, mb_doubles), axis=2)


    #Calculate centroid of each PE cycles count
    centroidList = []
    for cluster in points:
        centroidPoint = np.sum(cluster, axis=1) / cluster.shape[1]
        centroidList.append(centroidPoint)

    generate_csv_stdout(goldReferenceCycles, centroidList)

    # Plot all points and Centroids
    plt.figure()
    if (model_option == 'linear'):
        plt.title('Ideal Linear Regression Curve Parameters');
        plt.xlabel("slope (m)");
        plt.ylabel("y-intercept (b)");
    elif (model_option == 'exponential'):
        plt.title('Exponential Regression Curve Parameters');
        plt.xlabel("time-scale (b)");
        plt.ylabel("amplitude (a)");

    for cluster in points[0:10]:
        plt.plot(cluster[0], cluster[1], ls='', marker='o')

    plt.gca().set_prop_cycle(None)
    for centroid, label in zip(centroidList[0:10], goldReferenceCycles[0:10]):
        plt.plot(centroid[0], centroid[1], ls='', marker='*')
        plt.annotate(str(label / 1000)+'k', (centroid[0], centroid[1]))

    plt.show()


    ###############################################################################


def generate_csv_stdout(referenceCycles, centroidList):
    for reference, centroid in zip(referenceCycles, centroidList):
        print(str(reference) + ',' + str(centroid[0]) + ',' + str(centroid[1]))


if __name__ == "__main__":
    main()
