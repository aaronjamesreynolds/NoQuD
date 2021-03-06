import NoQuD.step_characteristic_solve.StepCharacteristicSolver as SC
import NoQuD.read_input_data.read_csv_input_file as R
import NoQuD.HomogeneousSolve.NodalSolve as HS
import numpy as np
import matplotlib.pyplot as plt

sig_t, sig_sin, sig_sout, sig_f, nu, chi, groups, cells, cell_size, assembly_map, material, assembly_size, \
assembly_cells = R.read_csv('assembly_info_test.csv')
slab = SC.StepCharacteristicSolver(sig_t, sig_sin, sig_sout, sig_f, nu, chi, groups, cells, cell_size, material)
slab.solve()

test = HS.NodalSolve(['assembly_info_test.csv', 'assembly_info_single_test.csv', 'assembly_info_single_test_b.csv'])
test.solve()
x = np.arange(0.0, 40., 40. / 256.0)
plt.plot(x, test.flux[0, 0, :], linestyle='--')
plt.plot(x, slab.flux_new[0][:])
plt.plot(x, test.flux[0, 1, :], linestyle='--')
plt.plot(x, slab.flux_new[1][:])
plt.xlabel('Position [cm]')
plt.ylabel('Flux [s^-1 cm^-2]')
plt.title('Neutron Flux')
plt.figlegend(['QD: fast', 'Transport: fast', 'QD: thermal', 'Transport: thermal'], loc='center')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlim(0, 40)
plt.savefig('flux.eps', format='eps', dpi=1200)
plt.show()


plt.plot(x, test.flux[0, 1, :])
plt.plot(x, slab.flux_new[1][:])
plt.xlabel('Position [cm]')
plt.ylabel('Flux [s^-1 cm^-2]')
plt.title('Thermal Neutron Flux')
plt.figlegend(['Homogeneous', 'Heterogeneous'])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlim(0, 40)
plt.savefig('thermal_flux.eps', format='eps', dpi=1200)
plt.show()
