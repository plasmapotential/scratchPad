#readMatlab.py
import scipy.io
matlab_file = '/home/tom/work/CFS/projects/CREATE/CarMa0NL_Shot10_v6.mat'
m = scipy.io.loadmat(matlab_file)
print(m['psi_tot_grid'])
print(m['j_zeta_grid'])
