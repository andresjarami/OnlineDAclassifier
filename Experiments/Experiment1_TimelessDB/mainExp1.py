import functionsExp1 as functionsExp1
import sys

## Input Variables


featureSet = int(sys.argv[1])
startPerson = int(sys.argv[2])
endPerson = int(sys.argv[3])
folder = str(sys.argv[4])
typeDatabase = str(sys.argv[5])
printResults = bool(int(sys.argv[6]))
shotStart = int(sys.argv[7])
samplesInMemory = int(sys.argv[8])
initialExpTime = int(sys.argv[9])
finalExpTime = int(sys.argv[10])
nameFile = folder + '_' + typeDatabase + '_FS_' + sys.argv[1] + '_sP_' + sys.argv[2] + '_eP_' + sys.argv[3] + \
           '_sStart_' + sys.argv[7] + '_inTime_' + sys.argv[9] + '_fiTime_' + sys.argv[10] + '.csv'

# featureSet = 1
# startPerson = 31 # Cote 20-36,EPN 31-60, Nina5 1-10
# endPerson = 60 # Cote 20-36,EPN 31-60, Nina5 1-10
# folder = 'resultsExample/example'
# typeDatabase = 'EPN'
# printResults = 1
# shotStart = 1
# samplesInMemory = 0
# initialExpTime = 1
# finalExpTime = 1
# nameFile = folder + '_' + typeDatabase + '_FS_' + str(featureSet) + '_sP_' + str(
#     startPerson) + '_eP_' + str(endPerson) + '_sStart_' + str(shotStart) + '_inTime_' + str(
#     initialExpTime) + '_fiTime_' + str(finalExpTime) + '.csv'

# Upload Data
dataMatrix, _, _, classes, peoplePriorK, _, numberShots, _, allFeatures, _ = functionsExp1.uploadDatabases(
    typeDatabase, featureSet)

# Evaluation
functionsExp1.evaluation2(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson,
                         endPerson, allFeatures, printResults, samplesInMemory, shotStart, initialExpTime,
                         finalExpTime, typeDatabase)
