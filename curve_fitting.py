import numpy as np

class curveFitData:
    cycles = 0
    m = 0.0
    b = 0.0
    l = 0.0
    a = 0.0
    type = 'empty'

def computeApproximate(testDivision, testTimes, goldDivision, goldTimes, goldCycles, model='linear'):
    goldTriples = np.zeros((len(goldCycles), 3))
    testDoubles = np.zeros((len(goldCycles), 2))

    if (model == 'exponential'):
        gold_l, gold_a = getDivisionExpoFit(goldDivision, goldTimes)
        goldTriples = np.column_stack((goldCycles, gold_l, gold_a))
        test_l, test_a = getDivisionExpoFit(testDivision, testTimes)
        testDoubles = np.column_stack((test_l, test_a))

    elif (model == 'linear'):
        gold_m, gold_b = getDivisionLinearFit(goldDivision, goldTimes)
        goldTriples = np.column_stack((goldCycles, gold_m, gold_b))
        test_m, test_b = getDivisionLinearFit(testDivision, testTimes)
        testDoubles = np.column_stack((test_m, test_b))

    testApproxCycles = []
    for test in testDoubles:
        testApproxCycles.append(getNearestPoint(test, goldTriples)[0])


    approxList = []
    goldList = []
    for testDouble, approx, goldTriple in zip(testDoubles, testApproxCycles, goldTriples):
        item = curveFitData()
        gold = curveFitData()
        item.type = model
        gold.type = model
        if (item.type == 'exponential'):
            item.l = testDouble[0]
            item.a = testDouble[1]
            gold.l = goldTriple[1]
            gold.a = goldTriple[2]
        elif (item.type == 'linear'):
            item.m = testDouble[0]
            item.b = testDouble[1]
            gold.m = goldTriple[1]
            gold.b = goldTriple[2]
        else:
            raise Exception('Invalid curve fit model')
        item.cycles = approx
        gold.cycles = goldTriple[0]
        approxList.append(item)
        goldList.append(gold)

    approxPoints = testDoubles
    goldPoints = goldTriples[:, 1:3]

    return approxList, goldList, approxPoints, goldPoints


def computeModulatedEstimation(testDivision, testTimes, goldDivision, goldTimes, goldCycles, model='linear'):
    goldTriples = np.zeros((len(goldCycles), 3))
    testDoubles = np.zeros((len(goldCycles), 2))
    #NOT COMPLETE

    if (model == 'exponential'):
        gold_l, gold_a = getDivisionExpoFit(goldDivision, goldTimes)
        goldTriples = np.column_stack((goldCycles, gold_l, gold_a))
        test_l, test_a = getDivisionExpoFit(testDivision, testTimes)
        testDoubles = np.column_stack((test_l, test_a))

    elif (model == 'linear'):
        gold_m, gold_b = getDivisionLinearFit(goldDivision, goldTimes)
        goldTriples = np.column_stack((goldCycles, gold_m, gold_b))
        test_m, test_b = getDivisionLinearFit(testDivision, testTimes)
        testDoubles = np.column_stack((test_m, test_b))

    testApproxCycles = []
    for test in testDoubles:
        testApproxCycles.append(getNearestPoint(test, goldTriples)[0])


    approxList = []
    goldList = []
    for testDouble, approx, goldTriple in zip(testDoubles, testApproxCycles, goldTriples):
        item = curveFitData()
        gold = curveFitData()
        item.type = model
        gold.type = model
        if (item.type == 'exponential'):
            item.l = testDouble[0]
            item.a = testDouble[1]
            gold.l = goldTriple[1]
            gold.a = goldTriple[2]
        elif (item.type == 'linear'):
            item.m = testDouble[0]
            item.b = testDouble[1]
            gold.m = goldTriple[1]
            gold.b = goldTriple[2]
        else:
            raise Exception('Invalid curve fit model')
        item.cycles = approx
        gold.cycles = goldTriple[0]
        approxList.append(item)
        goldList.append(gold)

    approxPoints = testDoubles
    goldPoints = goldTriples[:, 1:3]

    return approxList, goldList, approxPoints, goldPoints


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


def getTwoNearestModulatedEstimation(point, triples):
    i1, i2, distanceRatio = getTwoNearestIndexes(point, triples)
    cycles1 = float(triples[i1][0])
    cycles2 = float(triples[i2][0])
    estimation = cycles1 * distanceRatio + cycles2 * (1 - distanceRatio)


def getTwoNearestPoints(point, triples):
    firstIndex, secondIndex, _ = getTwoNearestIndexes(point, triples)
    return triples[firstIndex], triples[secondIndex]


def getTwoNearestIndexes(point, triples):
    nearestIndex = 0
    bestDistance1 = 999999.9;
    for i in range(len(triples)):
        distance = np.sqrt((triples[i][1] - point[0])**2 + (triples[i][2] - point[1])**2)
        if distance < bestDistance1:
            bestDistance1 = distance
            nearestIndex = i

    secondNearestIndex = 0
    bestDistance2 = 999999.9;
    for i in range(len(triples)):
        distance = np.sqrt((triples[i][1] - point[0])**2 + (triples[i][2] - point[1])**2)
        if (distance < bestDistance2) and (i != nearestIndex):
            bestDistance2 = distance
            secondNearestIndex = i

    distanceRatio = bestDistance2 / (bestDistance1 + bestDistance2)

    return nearestIndex, secondNearestIndex, distanceRatio


def getDivisionLinearFit(divisionData, time_values):
    divisionData = divisionData.transpose()
    m_values = []
    b_values = []
    for series in divisionData:
        m, b = computeSingleLinearFit(time_values, series)
        m_values.append(m)
        b_values.append(b)
    return m_values, b_values

def getDivisionExpoFit(divisionData, time_values):
    divisionData = divisionData.transpose()
    a_values = []
    lambda_values = []
    for i in range(divisionData.shape[0]):
        l, a = computeSingleExpoFit(time_values, divisionData[i])
        lambda_values.append(l)
        a_values.append(a)
    return lambda_values, a_values

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

def computeSingleExpoFit(x_values, y_values):
    #x_values = np.delete(x_values, np.s_[getFirstZeroIndex(y_values) + 1 :]) # delete values after zero
    #y_values = np.delete(y_values, np.s_[getFirstZeroIndex(y_values) + 1 :]) # delete values after zero
    x_values = np.delete(x_values, np.s_[5:]) #delete values after first 5
    y_values = np.delete(y_values, np.s_[5:])
    A = np.full((len(x_values), 2), 1.0)
    A[:, 0] = x_values
    B = np.log(y_values + 0.00000001)
    solution = np.linalg.inv(A.transpose() @ A) @ A.transpose() @ B
    return solution[0], np.exp(solution[1])
