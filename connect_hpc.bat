@echo off
setlocal

:: Default HPC connection details. Replace with your actual username and hostname.
set "HPC_USER=%~1"
if "%HPC_USER%"=="" set "HPC_USER=aantriksh.124259"

set "HPC_HOST=%~2"
if "%HPC_HOST%"=="" set "HPC_HOST=10.16.1.50"

:: Connect to HPC and execute the setup commands
echo Connecting to HPC (%HPC_USER%@%HPC_HOST%)...
echo After logging in, the gait_env will be automatically activated.

:: SSH command with remote execution
:: The -t flag forces pseudo-terminal allocation, which is necessary for interactive sessions
ssh -t %HPC_USER%@%HPC_HOST% "module load cuda; source /home/soft/anaconda3/etc/profile.d/conda.sh; conda activate gait_env; cd '/home/%HPC_USER%/Gait Analysis'; echo '--- Gait Analysis HPC Environment Active ---'; exec bash -l"
