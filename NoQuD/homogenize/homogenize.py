import numpy as np
from NoQuD.read_input_data import read_csv_input_file as read_csv
from NoQuD.step_characteristic_solve.StepCharacteristicSolver import *
import os


class HomogenizeAssembly:

    def __init__(self, filename):

        current_directory = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(current_directory, filename)

        self.sig_t, self.sig_sin, self.sig_sout, self.sig_f, self.nu, self.chi, self.groups, self.cells, self.cell_size\
            , self.assembly_map, self.material, self.assembly_size, self.assembly_cells = read_csv.read_csv(file_path)

        self.slab = StepCharacteristicSolver(self.sig_t, self.sig_sin, self.sig_sout, self.sig_f, self.nu, self.chi,
                                             self.groups, self.cells, self.cell_size, self.material)
        self.slab.solve()
        self.average_flux = np.zeros(2)
        self.discontinuity_factor_left = np.zeros(2)
        self.discontinuity_factor_right = np.zeros(2)


        self.sig_t_h = np.zeros(2)
        self.sig_sin_h = np.zeros(2)
        self.sig_sout_h = np.zeros(2)
        self.sig_f_h = np.zeros(2)
        self.eddington_factor_h = np.zeros(2)
        self.sig_r_h = np.zeros(2)

    def calculate_homogenized_nuclear_data(self):

        for group in xrange(self.groups):
            for cell in xrange(self.cells):
                self.sig_t_h[group] = self.sig_t_h[group] + self.sig_t[group][self.material[cell]]\
                                      * self.slab.flux_new[group][cell]/np.sum(self.slab.flux_new[group][:])
                self.sig_sin_h[group] = self.sig_sin_h[group] + self.sig_sin[group][self.material[cell]] \
                                        * self.slab.flux_new[group][cell] / np.sum(self.slab.flux_new[group][:])
                self.sig_sout_h[group] = self.sig_sout_h[group] + self.sig_sout[group][self.material[cell]] \
                                      * self.slab.flux_new[group][cell] / np.sum(self.slab.flux_new[group][:])
                self.sig_f_h[group] = self.sig_f_h[group] + self.sig_f[group][self.material[cell]] \
                                      * self.slab.flux_new[group][cell] / np.sum(self.slab.flux_new[group][:])
                self.eddington_factor_h[group] = self.eddington_factor_h[group] + \
                                                 self.slab.eddington_factors[group][cell] * \
                                                 self.slab.flux_new[group][cell] / np.sum(self.slab.flux_new[group][:])
                self.sig_r_h = self.sig_t_h - self.sig_sin_h

    def calculate_discontinuity_factors(self):

        for group in xrange(self.groups):
            self.average_flux[group] = np.mean(self.slab.flux_new[group][:])
            self.discontinuity_factor_left = self.slab.edge_flux[group][self.cells]/self.average_flux[group]
            self.discontinuity_factor_right = self.slab.edge_flux[group][0]/self.average_flux[group]

    def perform_homogenization(self):

        self.calculate_homogenized_nuclear_data()
        self.calculate_discontinuity_factors()


class HomogenizeGlobe:

    def __init__(self, single_assembly_input_files):
        # single_assembly_input_files should contain the assembly information file, followed by the assembly information
        # files for each unique assembly.
        self.single_assembly_input_files = single_assembly_input_files

        current_directory = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(current_directory, self.single_assembly_input_files[0])

        self.sig_t, self.sig_sin, self.sig_sout, self.sig_f, self.nu, self.chi, self.groups, self.cells, self.cell_size \
            , self.assembly_map, self.material, self.assembly_size, self.assembly_cells = read_csv.read_csv(file_path)

        # 'a' stands for assembly. The variables below will hold the homogenized nuclear data for each unique assembly.
        self.eddington_factor_a = np.zeros((self.groups, len(single_assembly_input_files)-1))
        self.sig_r_a = np.zeros((self.groups, len(single_assembly_input_files)-1))
        self.sig_t_a = np.zeros((self.groups, len(single_assembly_input_files)-1))
        self.sig_sin_a = np.zeros((self.groups, len(single_assembly_input_files)-1))
        self.sig_sout_a = np.zeros((self.groups, len(single_assembly_input_files)-1))
        self.sig_f_a = np.zeros((self.groups, len(single_assembly_input_files)-1))
        self.f_a = np.zeros((self.groups, len(single_assembly_input_files)-1))

        # 'g' stand for global. The variables below will hold the homogenized nuclear data for each node.
        self.eddington_factor_g = np.zeros((self.groups, len(self.assembly_map)))
        self.sig_r_g = np.zeros((self.groups, len(self.assembly_map)))
        self.sig_t_g = np.zeros((self.groups, len(self.assembly_map)))
        self.sig_sin_g = np.zeros((self.groups, len(self.assembly_map)))
        self.sig_sout_g = np.zeros((self.groups, len(self.assembly_map)))
        self.sig_f_g = np.zeros((self.groups, len(self.assembly_map)))
        self.f_g = np.zeros((self.groups, len(self.assembly_map)))

    # initialize matrices for global map based on dimension of single_assembly_input_files




if __name__ == '__main__':
    test = HomogenizeAssembly('assembly_info_single_test.csv')
    test.perform_homogenization()
    print test.sig_sin_h