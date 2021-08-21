#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=48
#PBS -l mem=185GB
#PBS -l walltime=48:00:00
#PBS -l software=python
#PBS -l wd

folder="results/EPN"
database="EPN"
for j in {1..1}
do
	for i in {31..60}
	do
	python3 mainExp1.py $j $i $i $folder $database 0 1 1 > EPN$PBS_JOBID.log &
done
done
wait
