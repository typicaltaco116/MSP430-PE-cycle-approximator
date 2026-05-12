#!/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    goldenFilename = ''
    testFilename = ''

    if len(sys.argv) > 2:
        goldenFilename = sys.argv[1]
        testFilename = sys.argv[2]
    else:
        print("Error: Not enough arguments")
        exit()

    goldData1, goldData2, goldData3, times = getDataCSV(goldenFilename)
    testData1, testData2, testData3, testTimes = getDataCSV(testFilename)

    goldReferenceCycles = range(10000, 210000, 10000)

    gold_m, gold_b = getDivisionLinearFit(goldData1, times)
    goldCurveTriples = np.column_stack((goldReferenceCycles, gold_m, gold_b))

    m, b = getDivisionLinearFit(testData1, testTimes)
    testCurveDoubles = np.column_stack((m, b))

    # Plot first 10 (m, b) points for test and golden
    plt.figure()
    plt.plot(m[0:10], b[0:10], ls='', marker='o')
    plt.plot(gold_m[0:10], gold_b[0:10], ls='', marker='*')
    plt.title('First 10 (m, b) points (PE cycles < 100k)')
    for i in range(10):
        annotationStr = str((i + 1) * 10) + 'k'
        plt.annotate(annotationStr, (gold_m[i], gold_b[i]))

    # Plot Last 10 (m, b) points for test and golden
    plt.figure()
    plt.plot(m[10:20], b[10:20], ls='', marker='o')
    plt.plot(gold_m[10:20], gold_b[10:20], ls='', marker='*')
    plt.title('Last 10 (m, b) points (PE cycles > 100k)')
    for i in range(10, 20):
        annotationStr = str((i + 1) * 10) + 'k'
        plt.annotate(annotationStr, (gold_m[i], gold_b[i]))


    # Plot all Segment Curves
    fig, axs = plt.subplots(nrows=5, ncols=4, layout='constrained')
    flat_axs = axs.ravel()

    for i in range(len(goldCurveTriples)):
        approx = getApproximateCycles(testCurveDoubles[i], goldCurveTriples)
        print('Approximate = ', approx, ', Actual = ', goldCurveTriples[i][0])
        # Plot the Golden plot most similar
        flat_axs[i].plot(times, goldData1[:, i], label=str(goldReferenceCycles[i]), marker='o')

        # Plot the input plot
        flat_axs[i].plot(testTimes, testData2[:, i], label=str(approx), marker='o')

        # Plot the regression line
        line_y = goldCurveTriples[i][1] * times + goldCurveTriples[i][2]
        flat_axs[i].plot(times[0:line_y[line_y > -0.5].size], line_y[line_y > -0.5])

        flat_axs[i].legend()
        flat_axs[i].grid(True)

    plt.show()

    ###############################################################################

def getApproximateCycles(point, goldenTriples):
    nearest = getNearestPoint(point, goldenTriples)
    return nearest[0]

def getNearestPoint(point, triples):
    return triples[getNearestIndex(point, triples)]

def getNearestIndex(point, triples):
    nearestIndex = 0
    bestDistance = 999999.9;
    for i in range(len(triples)):
        distance = np.sqrt((triples[i][1] - point[0])**2 + (triples[i][2] - point[1])**2)
        if distance < bestDistance:
            bestDistance = distance
            nearestIndex = i
    return nearestIndex

def getDivisionLinearFit(divisionData, time_values):
    divisionData = divisionData.transpose()
    m_values = []
    b_values = []
    for series in divisionData:
        m, b = computeSingleLinearFit(time_values, series)
        m_values.append(m)
        b_values.append(b)
    return m_values, b_values

def getFirstZeroIndex(array):
    value = 0
    while (value < len(array)) and (array[value] > 0):
        value += 1
    return value

def computeSingleLinearFit(x_values, y_values):
    #x_values = np.delete(x_values, np.s_[getFirstZeroIndex(y_values) + 1 :]) # delete values after zero
    #y_values = np.delete(y_values, np.s_[getFirstZeroIndex(y_values) + 1 :]) # delete values after zero
    x_values = np.delete(x_values, np.s_[5:]) #delete values after first 5
    y_values = np.delete(y_values, np.s_[5:])
    A = np.full((len(x_values), 2), 1.0)
    A[:, 0] = x_values
    B = y_values
    solution = np.linalg.inv(A.transpose() @ A) @ A.transpose() @ B
    return solution[0], solution[1]

def drawTable(data, rowLabels, columnLabels):
    stringData = []
    for row in data:
        stringData.append([f'{x:1.2f}' for x in row])
    fig = plt.figure()
    table = plt.table(cellText=stringData,
                      rowLabels=rowLabels,
                      colLabels=columnLabels,
                      loc='center')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)
    table.scale(1, 1.5)
    plt.show()

def drawDivisionPlots(division1Data, division2Data, division3Data, times):
    drawNewDivisionPlot(times, division1Data, 'Division 1')
    drawNewDivisionPlot(times, division2Data, 'Division 3')
    drawNewDivisionPlot(times, division3Data, 'Division 3')
    plt.show()


def drawNewDivisionPlot(times, data, title):
    stressingAmountsRow = range(10000, 220000, 10000)

    fig1 = plt.figure()
    for i in range(20):
        mylinestyle = 'dotted'
        if i >= 10:
            mylinestyle = 'solid'
        plt.plot(times, data[:, i], label=str(stressingAmountsRow[i]), 
                 marker='o', linestyle=mylinestyle)
    plt.xlabel('Partial Erase Times (ms)')
    plt.ylabel('Bit Error Rate (%)')
    plt.legend()
    plt.title(title)
    plt.grid(True)

def getDataCSV(filename):
    data = np.genfromtxt(filename, delimiter=',', encoding=None, dtype=str, 
                         skip_header=7, skip_footer=2)

    data = data.transpose()

    data = np.delete(data, 21, axis=0) # Delete the last row full of garbage

    # Typecast division data from csv
    division1Data = (data[1:, 1:21].astype(float) / 4096.0) * 100.0
    division2Data = (data[1:, 23:43].astype(float) / 4096.0) * 100.0
    division3Data = (data[1:, 45:65].astype(float) / 4096.0) * 100.0

    timesStr = data[1:, 0]
    for i in range(len(timesStr)):
        timesStr[i] = timesStr[i].replace('ms', '')

    times = timesStr.astype(float)

    if 'us' in data[0, 0]:
        times /= 1E3

    return division1Data, division2Data, division3Data, times

if __name__ == "__main__":
    main()
