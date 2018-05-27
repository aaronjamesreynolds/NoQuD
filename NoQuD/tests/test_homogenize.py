import NoQuD.homogenize.homogenize as homo
import os

def test_homogenize_assembly_running():

    """ Make sure the HomogenizeAssembly class can be initialized (which also performs the homogenization) without
        error. """

    test = homo.HomogenizeAssembly('test_files\\assembly_info_single_test.csv')


def test_homogenize_global_running():

    """ Make sure the HomogenizeGlobal class can be initialized (which also performs the homogenization) without
        error. """

    test = homo.HomogenizeGlobe(['testing_files\\assembly_info_test.csv', 'testing_files\\assembly_info_single_test.csv',
                                 'testing_files\\assembly_info_single_test.csv'])


def test_homogenize_assembly_correct_results():

    """ Make sure the HomogenizeAssembly class produces valid results. """

    test = homo.HomogenizeAssembly('test_files\\assembly_info_single_test.csv')


def test_homogenize_global_correct_results():

    """ Make sure the HomogenizeGlobal class produces valid results. """

    test = homo.HomogenizeGlobe(
        ['testing_files\\assembly_info_test.csv', 'testing_files\\assembly_info_single_test.csv',
         'testing_files\\assembly_info_single_test.csv'])



if __name__=="__main__":
    test_homogenize_assembly_running()
    test_homogenize_global_running()