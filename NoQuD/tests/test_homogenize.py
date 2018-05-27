import NoQuD.homogenize.homogenize as homo
import os

def test_homogenize_assembly_running():

    """ Make sure the HomogenizeAssembly class can be initialized (which also performs the homogenization) without
        error. """
    file_path = os.path.join('testing_files', 'assembly_info_single_test.csv')
    test = homo.HomogenizeAssembly(file_path)

def test_homogenize_global_running():

    """ Make sure the HomogenizeGlobal class can be initialized (which also performs the homogenization) without
        error. """

    test = homo.HomogenizeGlobe(['testing_files\\assembly_info_test.csv', 'testing_files\\assembly_info_single_test.csv',
                                 'testing_files\\assembly_info_single_test.csv'])

def test_homogenize_assembly_correct_results():

    print 'blah'

def test_homogenize_global_correct_results():

    print 'blah'

if __name__=="__main__":
    test_homogenize_assembly_running()
    test_homogenize_global_running()