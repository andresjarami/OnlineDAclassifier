import functionsExp1 as functionsExp1
import sys

## Input Variables


featureSet = int(sys.argv[1])
startPerson = int(sys.argv[2])
endPerson = int(sys.argv[3])
place = str(sys.argv[4])
typeDatabase = str(sys.argv[5])
printR = bool(int(sys.argv[6]))
shotStart = int(sys.argv[7])
samplesInMemory = int(sys.argv[8])
randomSeed = int(sys.argv[9])
expTimes = int(sys.argv[10])
nameFile = place + '_' + typeDatabase + '_FeatureSet_' + sys.argv[1] + '_startPerson_' + sys.argv[2] + '_endPerson_' + sys.argv[
    3] + '_shotStart_' + sys.argv[7]+ '_memmory_' + sys.argv[8] + '.csv'


# for shotStart in range(1,3):

# featureSet = 1
# startPerson = 23  # Cote 20-36,EPN 31-60, Nina5 1-10
# endPerson = 23  # Cote 20-36,EPN 31-60, Nina5 1-10
# place = 'resultsExample/exampleE'
# typeDatabase = 'Cote'
# printR = 1
# shotStart = 1
# samplesInMemory = 0
# randomSeed = 1
# expTimes = 2
# nameFile = place + '_' + typeDatabase + '_FeatureSet_' + str(featureSet) + '_startPerson_' + str(
#     startPerson) + '_endPerson_' + str(endPerson) + '_shotStart_' + str(shotStart) + '_memmory_' + str(samplesInMemory) + '.csv'



# Upload Data
dataMatrix, _, _, classes, peoplePriorK, _, numberShots, _, allFeatures, _ = functionsExp1.uploadDatabases(
    typeDatabase, featureSet)

# Evaluation
functionsExp1.evaluation(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson,
                         endPerson, allFeatures, typeDatabase, printR, samplesInMemory, shotStart, randomSeed, expTimes)
