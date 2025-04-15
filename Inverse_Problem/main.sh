# IDAS or Argon doesn't allow user to run .sh files via ".\xxx.sh" directly;
# So just copy them to terminal in Argon and run

# (optional)activate the virtual environment in python
cd ~/virtenvs/Hydro_py3108/bin/
source ./activate
cd ~/DA/2025_EKI/Inverse_Problem/

# Check qstat -u zli333 before submitting jobs, if there are remaining, use qdel to delete them
qstat -u zli333

# Check available slots before running code, if the slots available is less than num_ensemble * num_parallel_slots, then we need to set num_parallel_slots to be smaller; you should be able to monitor the job status also using this command
qstat -f -q IFC -u zli333

# Modify the parameters in `config.j2` as desired

# Submit the job to Argon to run
~/virtenvs/Hydro_py3108/bin/python ~/DA/2025_EKI/Inverse_Problem/eki_test.py ~/DA/2025_EKI/Inverse_Problem/config.j2 