import numpy as np
from NoQuD.read_input_data import read_csv_input_file as read_csv
from NoQuD.step_characteristic_solve.StepCharacteristicSolver import *

class Homogenize():

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
             self.discontinuity_factor_left = self.slab.edge_flux[group][self.cells+1]/self.average_flux[group]
             self.discontinuity_factor_right = self.slab.edge_flux[group][0]/self.average_flux[group]
