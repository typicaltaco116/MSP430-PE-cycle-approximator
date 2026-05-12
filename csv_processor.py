import numpy as np

def getDataCSV(filename, divisionSizes, divisionStarts, segmentSize=4096.0):
    data = np.genfromtxt(filename, delimiter=',', encoding=None, dtype=str, 
                         skip_header=7, skip_footer=2)

    data = data.transpose()

    data = np.delete(data, 21, axis=0) # Delete the last row full of garbage

    if (len(divisionSizes) != len(divisionStarts)):
        raise Exception("Unequal division parameters")

    divisionData = []
    for start, size in zip(divisionStarts, divisionSizes):
        divisionData.append((data[1:, start:start+size].astype(float) / segmentSize) * 100.0)

    timesStr = data[1:, 0]
    for i in range(len(timesStr)):
        timesStr[i] = timesStr[i].replace('ms', '')

    times = timesStr.astype(float)

    if 'us' in data[0, 0]:
        times /= 1E3

    return divisionData, times

def get_CSV_zeroSegments(filename, segments, segmentSize=4096.0):
    data = np.genfromtxt(filename, delimiter=',', encoding=None, dtype=str, 
                         skip_header=7, skip_footer=2)
    data = data.transpose()

    data = np.delete(data, 21, axis=0) # Delete the last row full of garbage

    segmentData = []
    for segment in segments:
        segmentData.append((data[1:, segment + 1].astype(float) / segmentSize) * 100.0)

    timesStr = data[1:, 0]
    for i in range(len(timesStr)):
        timesStr[i] = timesStr[i].replace('ms', '')

    times = timesStr.astype(float)

    if 'us' in data[0, 0]:
        times /= 1E3

    return segmentData, times
