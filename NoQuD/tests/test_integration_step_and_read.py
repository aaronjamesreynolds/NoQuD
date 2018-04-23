from NoQuD.read_input_data import read_csv_input_file as read_csv
from NoQuD.StepCharacteristicSolve.StepCharacteristicSolver import *
import os

def test_integration():

    current_directory = os.path.abspath(__file__)
    file_path = os.path.join(current_directory, '../AI_test.csv')

    sig_t, sig_sin, sig_sout, sig_f, nu, chi, groups, cells, cell_size, assembly_map, material, \
    assembly_size, assembly_cells = read_csv.read_csv(file_path)

    slab = StepCharacteristicSolver(sig_t, sig_sin, sig_sout, sig_f, nu, chi)

if __name__ == '__main__':

    test_integration()