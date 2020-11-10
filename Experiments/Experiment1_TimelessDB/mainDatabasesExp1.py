import sys
import functionsExp1 as functionsExp1

## Input Variables

# featureSet = int(sys.argv[1])
# startPerson = int(sys.argv[2])
# endPerson = int(sys.argv[3])
# place = str(sys.argv[4])
# typeDatabase = str(sys.argv[5])
# pca = bool(int(sys.argv[6]))
# printR = bool(int(sys.argv[7]))
# hyper = bool(int(sys.argv[8]))
# eval = bool(int(sys.argv[9]))
# nameFile = place + '_FeatureSet_' + sys.argv[1] + '_startPerson_' + sys.argv[2] + '_endPerson_' + sys.argv[3] + '.csv'

# typeDatabaseSet = ['Cote']
typeDatabaseSet = ['EPN']
for typeDatabase in typeDatabaseSet:

    featureSet = 1
    startPerson = 1
    endPerson = 30
    pca = bool(1)
    printR = bool(1)
    hyper = bool(0)
    eval = bool(1)
    nameFile = typeDatabase+'_Uns_2Shots.csv'

    shotStart=2

    # Upload Data
    dataMatrix, _, _, classes, peoplePriorK, _, numberShots, _, allFeatures, _ = functionsExp1.uploadDatabases(
        typeDatabase, featureSet)



    # Evaluation

    if eval:
        functionsExp1.evaluation(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson, endPerson,
                                 allFeatures, typeDatabase, printR,shotStart)
