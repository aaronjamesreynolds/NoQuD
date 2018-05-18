import numpy as np
from NoQuD.homogenize.homogenize import *
from NoQuD.nodal.Nodal import *
from numpy.polynomial.legendre import legval as legval


class NodalSolve:

    """ Creates homogenized data for a 1D, 2 group problem and solves using nodal quasi-diffusion methods.

    The homogenized nuclear data for the assembly(s) described by the input files is calculated. From this homogenized
    data, the linear system required for a nodal solve is generated. Variables necessary for the solve process are
    initialized.

    Args:
        assembly_input_files:  should contain the assembly information file, followed by the assembly information
        files for each unique assembly.

    Attributes:
        homogenized_problem (HomogenizeGlobe instance): contains homogenized nuclear data.
        assembly_sizes (float[][]): size of each unique assembly in each group
        nodal_data (Nodal instance): contains the linear system necessary for the nodal solve method
        order (int): number of coefficients in polynomial approximation to flux in each node
        groups (int): number of energy groups in problem (current implementation only works for two groups)
        nodes (int): number of nodes in problem. Equal to number of assemblies present in heterogeneous problem.
        solution_cells (int): mesh length in homogeneous solution. This is a per node quantity.
        fast_system (float[][]): the linear system used to solve for the fast flux.
        slow_system (float[][]): the linear system used to solve for the slow, or thermal, flux.
        fission_source (float[][]): the fission source terms in the RHS of the nodal solve
        scatter_source (float[][]): the scatter source terms in the RHS of the nodal solve.
        k (float[]): the multiplication factor. Stores the current value and two previous values.
        flux_coefficients (float[][]): coefficients to the polynomial flux approximation
        flux (float[][][]): the fast and thermal fluxes across the nodes. Stores and current and previous values.
        spatial_fission_source (float[][][]): the fission source across the nodes. Stores the current and previous
            values.
        chi (float[]): the probability for a fission neutron to be born into a particular energy group.
        nu (float[]): the average number of neutrons produced per fission in each energy group.
        flux_epsilon (float): a convergence parameter on the flux
        k_epsilon (float): a convergence parameter on the mulitplication factor.
        converged (boolean): evaluates whether all convergence parameters are met.
        k_converged (boolean): evaluation of multiplication factor convergence.
        flux_converged (boolean): evaluation of flux convergence.
        fission_source_converged (boolean): evaluation of fission source convergence.
    """

    def __init__(self, assembly_input_files):

        # Obtain homogenized data
        self.homogenized_problem = HomogenizeGlobe(assembly_input_files)

        # Obtain linear system for nodal solve
        self.assembly_sizes = self.homogenized_problem.assembly_size*np.ones((self.homogenized_problem.groups,
                                                                          len(self.homogenized_problem.assembly_map)))
        eddington_over_sig_t = np.divide(self.homogenized_problem.eddington_factor_g, self.homogenized_problem.sig_t_g)
        self.nodal_data = Nodal(eddington_over_sig_t, self.homogenized_problem.sig_r_g,
                                self.assembly_sizes, self.homogenized_problem.f_g,
                                self.homogenized_problem.groups, len(self.homogenized_problem.assembly_map))

        # Assign commonly used variables in from HomogenizeGlobe and Nodal instances.
        self.order = self.nodal_data.order_of_legendre_poly
        self.groups = self.homogenized_problem.groups
        self.nodes = self.nodal_data.nodes
        self.solution_cells = self.homogenized_problem.cells/self.nodes # this is a per node quantity
        self.fast_system = self.nodal_data.linear_system[0, :, :]
        self.slow_system = self.nodal_data.linear_system[1, :, :]

        # Initialize variables.
        self.fission_source = np.zeros((self.groups,
                                        self.nodes * self.order))
        self.scatter_source = np.zeros((self.groups,
                                        self.nodes * self.order))
        self.k = [1.0, 2.0, 1.2]
        self.flux_coefficients = np.ones((self.groups, self.nodes * self.order))
        self.flux = np.ones((3, self.groups, self.nodes*self.solution_cells))
        self.flux[2, :, :] = 2*np.ones((self.groups, self.nodes*self.solution_cells))
        self.spatial_fission_source = np.ones((2, self.groups, self.nodes*self.solution_cells))

        # Explicitly define chi and nu, as presently I haven't looked into homogenizing them.
        self.chi = np.zeros((self.groups, 1))
        self.chi[0] = self.homogenized_problem.chi[0, 0]
        self.chi[1] = self.homogenized_problem.chi[1, 0]
        self.nu = np.zeros((self.groups, 1))
        self.nu[0] = self.homogenized_problem.nu[0, 0]
        self.nu[1] = self.homogenized_problem.nu[1, 0]
        self.flux_epsilon = 0
        self.k_epsilon = 0
        self.converged = False
        self.k_converged = False
        self.flux_converged = False
        self.fission_source_converged = False

        # Not functional for more than two groups.
    def form_fission_source(self):
        for node in xrange(self.nodes):
            for eqn in xrange(self.order - 2):
                for group in xrange(self.groups):
                    fsi = 2*self.nodes+self.nodes*eqn + node  # Fission Source Index (fsi)
                    fci = eqn + node*self.order  # Flux Coefficient Index (fci)
                    self.fission_source[1 - group, fsi] = self.chi[1 - group] * self.nu[group] * \
                                                          self.homogenized_problem.sig_f_g[group, node] * \
                                                          self.flux_coefficients[group, fci] / self.k[1]

    def form_scatter_source(self):
        for node in xrange(self.nodes):
            for eqn in xrange(self.order - 2):
                for group in xrange(self.groups):
                    ssi = 2*self.nodes+self.nodes*eqn + node  # Scatter Source Index (fsi)
                    fci = eqn + node*self.order  # Flux Coefficient Index (fci)
                    self.scatter_source[1 - group, ssi] = self.homogenized_problem.sig_sout_g[group][node] * \
                                                          self.flux_coefficients[group, fci]

    def iterate_flux_coefficients(self):

        inverse = np.linalg.inv(self.fast_system)
        self.flux_coefficients[0, :] = np.dot(inverse, np.array([self.fission_source[0, :]]).T)[:, 0]
        inverse = np.linalg.inv(self.slow_system)
        self.flux_coefficients[1, :] = np.dot(inverse, np.array([self.scatter_source[1, :]]).T)[:, 0]

    def build_spatial_flux(self):

        for node in xrange(self.nodes):
            for index in xrange(self.solution_cells):
                normalized_index = self.normalized_index(index)
                flux_index = self.solution_cells*node+index
                coeff_start = self.order*node
                coeff_end = self.order*(node+1)
                self.flux[0, 1, flux_index] = legval(normalized_index, self.flux_coefficients[1, coeff_start:coeff_end])
                self.flux[0, 0, flux_index] = legval(normalized_index, self.flux_coefficients[0, coeff_start:coeff_end])

    def build_spatial_fission_source(self):

        for node in xrange(self.nodes):
            for group in xrange(self.groups):
                for index in xrange(self.solution_cells):
                    si = node*self.solution_cells + index  # Source Index (si)
                    self.spatial_fission_source[0, 1-group, si] = self.chi[1-group] * self.nu[group] *\
                                                                   self.homogenized_problem.sig_f_g[group, node] * self.flux[0, group, si]

    def iterate_eigenvalue(self):

        self.k[0] = self.k[1]*np.sum(self.spatial_fission_source[0, :, :])/np.sum(self.spatial_fission_source[1, :, :])

    def normalized_index(self, unnormalized_index):

        normalized_index = (2*unnormalized_index-self.solution_cells)/float(self.solution_cells)+1/self.solution_cells
        return normalized_index

    def calculate_convergence_parameters(self):

        current_gen_diff = np.abs(self.flux[0, :, :]-self.flux[1, :, :])
        last_gen_diff = np.abs(self.flux[1, :, :]-self.flux[2, :, :])
        self.flux_epsilon = np.amax(current_gen_diff)/np.amax(last_gen_diff)
        current_gen_diff = np.abs(self.k[0] - self.k[1])
        last_gen_diff = np.abs(self.k[1] - self.k[2])
        self.k_epsilon = current_gen_diff / last_gen_diff

    def evaluate_convergence_criteria(self):

        self.k_converged = np.amax(np.abs(self.k[0] - self.k[1]))/np.abs(1-self.k_epsilon) < 1e-6
        self.flux_converged = np.amax(np.abs(self.flux[0,0,:] - self.flux[1,0,:]))/np.abs(1-self.flux_epsilon) < 1e-6
        self.fission_source_converged = np.amax(np.abs(self.spatial_fission_source[0,0,:]
                                                       - self.spatial_fission_source[1,0,:])) < 1e-6

    def solve(self):

        while not self.converged:
            self.form_fission_source()
            self.iterate_flux_coefficients()
            self.form_scatter_source()
            self.iterate_flux_coefficients()
            self.build_spatial_flux()
            self.build_spatial_fission_source()
            self.iterate_eigenvalue()
            self.calculate_convergence_parameters()
            self.evaluate_convergence_criteria()
            if self.k_converged and self.flux_converged and self.fission_source_converged:
                self.converged = True
                integral_flux = np.sum(self.flux[0,:,:])
                self.flux[0, :, :] = self.flux[0, :, :]/integral_flux  # normalize flux
                print self.k[0]
            else:
                self.k[2] = self.k[1]
                self.k[1] = self.k[0]
                self.flux[2, :, :] = self.flux[1, :, :]
                self.flux[1, :, :] = self.flux[0, :, :]
                self.spatial_fission_source[1, :, :] = self.spatial_fission_source[0, :, :]

if __name__=="__main__":

    test = NodalSolve(['assembly_info_test.csv', 'assembly_info_single_test.csv', 'assembly_info_single_test_b.csv'])
    test.solve()
    x = numpy.arange(0.0, 30., 30.0 / 384.0)
    import matplotlib.pyplot as plt
    plt.plot(x, test.flux[0, 0, :])
    plt.plot(x, test.flux[0, 1, :])
    plt.xlabel('Position [cm]')
    plt.ylabel('Flux [s^-1 cm^-2]')
    plt.title('Neutron Flux')
    plt.show()





