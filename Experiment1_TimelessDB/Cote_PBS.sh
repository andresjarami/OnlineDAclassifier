#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=40
#PBS -l mem=50GB
#PBS -l walltime=48:00:00
#PBS -l software=python
#PBS -l wd

folder="results/Cote"
database="Cote"
for j in {1..1}
do
	for i in {20..36}
	do
	python3 mainExp1.py $j $i $i $folder $database 0 1 1 > Cote$PBS_JOBID.log &
done
done
wait
