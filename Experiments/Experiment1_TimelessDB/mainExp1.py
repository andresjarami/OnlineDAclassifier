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
nameFile = place + '_FeatureSet_' + sys.argv[1] + '_startPerson_' + sys.argv[2] + '_endPerson_' + sys.argv[
    3] + 'shotStart' + sys.argv[7] + '.csv'

# featureSet = 1
# startPerson = 21  # Cote 20-36,EPN 31-60, Nina5 1-10
# endPerson = 21  # Cote 20-36,EPN 31-60, Nina5 1-10
# place = 'example'
# typeDatabase = 'Cote'
# printR = 1
# shotStart = 2
# nameFile = place + '_' + typeDatabase + '_FeatureSet_' + str(featureSet) + '_startPerson_' + str(
#     startPerson) + '_endPerson_' + str(endPerson) + 'shotStart' + str(shotStart) + '.csv'

# # typeDatabaseSet = ['Cote']
# typeDatabaseSet = ['Cote', 'Nina5']
# for typeDatabase in typeDatabaseSet:
#
#     if typeDatabase == 'Nina5':
#         startPerson = 1  # Cote 20-36,EPN 31-60, Nina5 1-10
#         endPerson = 10
#     elif typeDatabase == 'Cote':
#         startPerson = 20  # Cote 20-36,EPN 31-60, Nina5 1-10
#         endPerson = 36
#
#     # startPerson = 2 #Cote 20-36,EPN 31-60, Nina5 1-10
#     # endPerson = 2
#     featureSet = 1
#     printR = bool(1)
#     shotStart = 1
#     nameFile = typeDatabase + '_FeatureSet_' + str(featureSet) + '_startPerson_' + str(
#         startPerson) + '_endPerson_' + str(endPerson) + '_shotStart_' + str(shotStart) + '.csv'
#     # nameFile = None

# Upload Data
dataMatrix, _, _, classes, peoplePriorK, _, numberShots, _, allFeatures, _ = functionsExp1.uploadDatabases(
    typeDatabase, featureSet)

# Evaluation
functionsExp1.evaluation(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson,
                         endPerson,
                         allFeatures, typeDatabase, printR, shotStart)
