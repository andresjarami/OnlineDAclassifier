import functions_exp as functions_exp
import sys

#%% Input Variables

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


# %%## Example of the parameters
'''
# featureSet 1 (logVar) 2 (MAV,WL,ZC,SSC) or 3(LS,MFL,MSR,WAMP)
featureSet = 1
# typeDatabase: 'Nina5' or 'Cote' or 'EPN_120' or 'Capgmyo_dbb' or 'LongTerm3DC 1-20'
typeDatabase = 'Cote'
# people Nina5 (1-10) Cote(20-36) EPN_120(1-120) Capgmyo_dbb(1-10) LongTerm3DC(1-20)
startPerson = 1
endPerson = 36
folder = '../Results/example'
# printR: Print results (True or False)
printResults = 1
# shotStart: how many gestures per class are in the initial training set
shotStart = 1
# initialExpTime and finalExpTime are how many time the experiment is run and specify the seeds used
initialExpTime = 1
finalExpTime = 20
# nameFile: Put in string the name of the file where you want to save the results or None to not save a file
nameFile = folder + '_' + typeDatabase + '_FS_' + str(featureSet) + '_sP_' + str(
    startPerson) + '_eP_' + str(endPerson) + '_sStart_' + str(shotStart) + '_inTime_' + str(
    initialExpTime) + '_fiTime_' + str(finalExpTime) + '.csv'
'''


#%% Evaluation

functions_exp.evaluation(featureSet, nameFile, startPerson, endPerson, printResults, shotStart, initialExpTime,
                     finalExpTime, typeDatabase,all_models = True)



