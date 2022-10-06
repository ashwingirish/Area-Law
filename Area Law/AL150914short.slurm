#!/bin/bash
#SBATCH --job-name=Job150914
#SBATCH --partition=long 
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=ashwin.girish@icts.res.in
#SBATCH --ntasks=1
#SBATCH --nodes=4
#SBATCH --mem=1gb     
#SBATCH --output=serial_test_%j.log   
##Standard output and error log
pwd; hostname; date

module load python
export LAL_DATA_PATH=/nasdata1/ashwin.girish
python AL_GW150914short.py
date
