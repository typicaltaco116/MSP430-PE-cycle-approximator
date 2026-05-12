#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import operator
import sys

from csv_processor import getDataCSV, get_CSV_zeroSegments
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
                model_option = arg
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

    testData, testTimeSeries = getDataCSV(filenameList.pop(0), divisionSizes, startIndexes)
    testDivision = testData[1]

    goldReferenceCycles = range(0, 210000, 10000)
    # For S6 ONLY!!!
    testReferenceCycles = np.repeat(np.arange(0, 100000, 10000), 2) + np.tile(np.array([8000, 12000]), 10)

    #Open and gather data from all files
    for file in filenameList:
        divisionData, timeSeries = getDataCSV(file, divisionSizes, startIndexes)

        # Add zero segments
        zeroSegments, _ = get_CSV_zeroSegments(file, [20, 21, 42])
        divisionData[0] = np.concatenate((zeroSegments[0].reshape(-1, 1), divisionData[0]), axis=1)
        divisionData[1] = np.concatenate((zeroSegments[1].reshape(-1, 1), divisionData[1]), axis=1)
        divisionData[2] = np.concatenate((zeroSegments[2].reshape(-1, 1), divisionData[2]), axis=1)

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

        # Build rest of list
        for BER_series, time_series in zip(divisionSeriesList[1:], timeSeriesList[1:]):
            m_values, b_values = getDivisionLinearFit(BER_series, time_series)
            mb_doubles = np.column_stack((m_values, b_values))
            mb_doubles = mb_doubles[:, :, np.newaxis]
            points = np.concatenate((points, mb_doubles), axis=2)

        # Get testPoints
        m_values, b_values = getDivisionLinearFit(testDivision, testTimeSeries)
        testPoints = np.column_stack((m_values, b_values))

    else:
        # Initial sizing of points list
        l_values, a_values = getDivisionExpoFit(divisionSeriesList[0], timeSeriesList[0])
        points = np.column_stack((l_values, a_values))
        points = points[:, :, np.newaxis]

        # Build rest of list
        for BER_series, time_series in zip(divisionSeriesList[1:], timeSeriesList[1:]):
            l_values, a_values = getDivisionLinearFit(BER_series, time_series)
            mb_doubles = np.column_stack((l_values, a_values))
            mb_doubles = mb_doubles[:, :, np.newaxis]
            points = np.concatenate((points, mb_doubles), axis=2)

        # Get testPoints
        l_values, a_values = getDivisionExpoFit(testDivision, testTimeSeries)
        testPoints = np.column_stack((l_values, a_values))
    

    #Calculate centroid of each PE cycles count
    centroidList = []
    for cluster in points:
        centroidPoint = np.sum(cluster, axis=1) / cluster.shape[1]
        centroidList.append(centroidPoint)

    #generate_csv_stdout(goldReferenceCycles, centroidList)


    approx = []
    for srcPoint in testPoints:
        nPoints = 5
        sortedList = get_closest_points(srcPoint, points, goldReferenceCycles, nPoints)
        sortedArray = np.array(sortedList)
        reverseWeight = sortedArray[nPoints - 1, 1] - sortedArray[:, 1]
        normalized = reverseWeight / np.sum(reverseWeight)
        avg = np.sum(sortedArray[:, 0] * normalized)
        approx.append(avg)

    for estimatedValue, actualValue in zip(approx, testReferenceCycles):
        print("estimated = " + str(int(estimatedValue)) + " actual = " + str(actualValue))



    # Plot all points and Centroids
    plt.figure()
    plt.title(model_option + ' curve parameters plane division points')
    for cluster in points:
        plt.plot(cluster[0], cluster[1], ls='', marker='o')

    plt.gca().set_prop_cycle(None)
    for centroid, label in zip(centroidList, goldReferenceCycles):
        plt.plot(centroid[0], centroid[1], ls='', marker='*')
        plt.annotate(str(label / 1000)+'k', (centroid[0], centroid[1]))

    plt.plot(testPoints[:, 0], testPoints[:, 1], ls='', marker='+', color='black')

    plt.show()


    ###############################################################################


def generate_csv_stdout(referenceCycles, centroidList):
    for reference, centroid in zip(referenceCycles, centroidList):
        print(str(reference) + ',' + str(centroid[0]) + ',' + str(centroid[1]))

def get_distance(p1, p2):
    accumulator = 0.0
    for n1, n2 in zip(p1, p2):
        accumulator += (n1 - n2)**2
    return np.sqrt(accumulator)

def get_closest_points(src, points, refCycles, return_length): # points is a 3D array
    distanceList = []
    for cluster, cycles in zip(points, refCycles):
        for point in cluster.transpose():
            distanceList.append([cycles, get_distance(src, point)])
    distanceList.sort(key=operator.itemgetter(1)) # Sort list descending by distance
    return distanceList[0:return_length]


if __name__ == "__main__":
    main()
