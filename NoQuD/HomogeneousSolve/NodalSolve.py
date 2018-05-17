import numpy as np
from NoQuD.homogenize.homogenize import *
from NoQuD.nodal.Nodal import *


class NodalSolve:


    def __init__(self, assembly_input_files):
        # single_assembly_input_files should contain the assembly information file, followed by the assembly information
        # files for each unique assembly.

        self.homogenized_problem = HomogenizeGlobe(assembly_input_files)
        self.assembly_cell_sizes = self.homogenized_problem.cell_size*np.ones((self.homogenized_problem.groups,
                                                                          len(self.homogenized_problem.assembly_map)))
        self.nodal_data = Nodal(self.homogenized_problem.eddington_factor_g, self.homogenized_problem.sig_r_g,
                                self.assembly_cell_sizes, self.homogenized_problem.f_g,
                                self.homogenized_problem.groups, len(self.homogenized_problem.assembly_map))
        self.order = self.nodal_data.order_of_legendre_poly
        self.groups = self.homogenized_problem.groups
        self.nodes = self.nodal_data.nodes
        self.linear_system = self.nodal_data.linear_system
        self.fast_system = self.linear_system[0, :, :]
        self.slow_system = self.linear_system[1, :, :]
        self.fission_source_old = np.zeros((self.homogenized_problem.groups,
                                            self.nodes * self.order))
        self.scatter_source_old = np.zeros((self.homogenized_problem.groups,
                                            self.nodes * self.order))
        self.k_old = 1.0
        self.order = self.nodal_data.order_of_legendre_poly
        self.flux_coefficients = np.ones((self.groups, self.nodes * self.order))

        # Explicitly define chi and nu, as presently I haven't looked into homogenizing them.
        self.chi = np.zeros((self.groups, 1))
        self.chi[0] = self.homogenized_problem.chi[0, 0]
        self.chi[1] = self.homogenized_problem.chi[1, 0]
        self.nu = np.zeros((self.groups, 1))
        self.nu[0] = self.homogenized_problem.nu[0, 0]
        self.nu[1] = self.homogenized_problem.nu[1, 0]

    # Not functional for more than two groups.
    def form_fission_source(self):
        for node in xrange(self.nodes):
            for eqn in xrange(self.order - 2):
                for group in xrange(self.groups):
                    fsi = 2*self.nodes+self.nodes*eqn + node  # Fission Source Index (fsi)
                    fci = eqn + node*self.order  # Flux Coefficient Index (fci)
                    self.fission_source_old[1-group, fsi] = self.chi[1-group] * self.nu[group] *\
                                                          self.homogenized_problem.sig_f_g[group, node] * \
                                                          self.flux_coefficients[group, fci] / self.k_old

    def form_scatter_source(self):
        for node in xrange(self.nodes):
            for eqn in xrange(self.order - 2):
                for group in xrange(self.groups):
                    ssi = 2*self.nodes+self.nodes*eqn + node  # Scatter Source Index (fsi)
                    fci = eqn + node*self.order  # Flux Coefficient Index (fci)
                    self.scatter_source_old[1-group, ssi] = self.homogenized_problem.sig_sout_g[group][node] * \
                                                            self.flux_coefficients[group, fci]

    def calculate_flux(self):

        inverse = np.linalg.inv(self.fast_system)
        self.flux_coefficients[0, :] = np.dot(inverse, np.array([self.fission_source_old[0, :]]).T)[:, 0]
        inverse = np.linalg.inv(self.slow_system)
        self.flux_coefficients[1, :] = np.dot(inverse, np.array([self.scatter_source_old[1, :]]).T)[:, 0]


if __name__=="__main__":

    test = NodalSolve(['assembly_info_test.csv', 'assembly_info_single_test.csv', 'assembly_info_single_test.csv'])
    test.form_fission_source()
    test.form_scatter_source()
    print test.scatter_source_old
    test.calculate_flux()
    print test.flux_coefficients