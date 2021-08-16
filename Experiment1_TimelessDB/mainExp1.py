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
initialExpTime = int(sys.argv[8])
finalExpTime = int(sys.argv[9])
nameFile = folder + '_' + typeDatabase + '_FS_' + sys.argv[1] + '_sP_' + sys.argv[2] + '_eP_' + sys.argv[3] + \
           '_sStart_' + sys.argv[7] + '_inTime_' + sys.argv[8] + '_fiTime_' + sys.argv[9] + '.csv'

# for featureSet in [1]:
#     folder = 'resultsExample/all'
#     # for info in [['Capgmyo_dbb', 1, 5],['Cote', 20, 24],['EPN_612', 1, 5] ]:
#     for info in [['Capgmyo_dbb', 1, 10]]:
#         typeDatabase = info[0]
#         startPerson = info[1]  # Cote 20-36,EPN_612 1-612, Nina5 1-10, LongTerm3DC 1-20, Capgmyo_dbb 1-10
#         endPerson = info[2]
#         printResults = 1
#         shotStart = 1 # LongTerm3DC 3(all of day 1),
#         initialExpTime = 1
#         finalExpTime = 3
#         nameFile = folder + '_' + typeDatabase + '_FS_' + str(featureSet) + '_sP_' + str(
#             startPerson) + '_eP_' + str(endPerson) + '_sStart_' + str(shotStart) + '_inTime_' + str(
#             initialExpTime) + '_fiTime_' + str(finalExpTime) + '.csv'

### Evaluation
functionsExp1.evaluation_notest(featureSet, nameFile, startPerson, endPerson, printResults, shotStart, initialExpTime,
                                finalExpTime, typeDatabase)

# functionsExp1.evaluation(featureSet, nameFile, startPerson, endPerson, printResults, shotStart,
#                                 initialExpTime,
#                                 finalExpTime, typeDatabase)

# functionsExp1.evaluation_Supervised(featureSet, nameFile, startPerson, endPerson, printResults, shotStart, initialExpTime,
#                          finalExpTime, typeDatabase)
