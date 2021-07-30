#!/bin/bash
#PBS -P oj72
#PBS -q hugemem
#PBS -l ncpus=48
#PBS -l mem=440GB
#PBS -l walltime=48:00:00
#PBS -l software=python
#PBS -l wd
 
folder="Results/EPN"
database="EPN"
for j in {1..3}
do

	for i in {1..30}
	do
		python3 main.py $j $i $i $folder $database 1 0 0 1 > EPN$PBS_JOBID.log &
	done
done
wait
