#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=48
#PBS -l mem=20GB
#PBS -l walltime=48:00:00
#PBS -l software=python
#PBS -l wd

folder="results/Nina5"
database="Nina5"
for j in {1..1}
do
	for i in {1..10}
	do
	python3 mainExp1.py $j $i $i $folder $database 0 1 1 > Nina5$PBS_JOBID.log &
done
done
wait
