# Import pandas
import pandas as pd
import numpy as np

def read_csv(filename):

    # Load csv
    data = pd.read_csv(filename, header = None)

    # Assign singular values from csv.
    assembly_types = int(data.iloc[0, 1])  # number of assembly types.
    assemblies = int(data.iloc[1, 1])  # number of total assemblies
    assembly_size = int(data.iloc[2, 1])  # size of assemblies
    cells = int(data.iloc[3, 1])  # number of cells
    groups = int(data.iloc[4, 1])  # number of neutron energy groups
    unique_materials = int(data.iloc[5, 1])  # number of unique materials

    # Initialize matrices to hold nuclear data in each unique material in each energy group
    sig_t = np.zeros([groups, unique_materials])  # total macro cross section (mcs)
    sig_sin = np.zeros([groups, unique_materials])  # in-scatter mcs
    sig_sout = np.zeros([groups, unique_materials])  # out-scatter mcs
    sig_f = np.zeros([groups, unique_materials])  # fission mcs
    nu = np.zeros([groups, unique_materials])  # avg. number of neutrons generated per fission
    chi = np.zeros([groups, unique_materials])  # probability for a fission neutron to appear in an energy group

    # Intialize matrices and assign geometry parameters.
    assembly_cells = int(cells/assemblies)  # the number of cells per assembly.
    key_length = np.zeros([1, assembly_types])  # the key length describes the length of periodicity in the material map
    local_map = np.zeros([assembly_types, assembly_cells])  # material map for each assembly
    assembly_map = np.zeros([1, assemblies])  # coarse map for order of assemblies
    cell_size = assemblies * assembly_size / cells  # length of each cell


    for i in xrange(0, assembly_types):
        key_length[0][i] = data.iloc[20 + 4 * i, 1]

    # This nest of for loops assigns the nuclear data given in the csv. Fast groups have
    # lower indexes
    for i in xrange(0, groups):  # loop over groups
        for j in xrange(0, unique_materials):  # loop over unique materials
            sig_t[i][j] = data.iloc[9, 1 + i + unique_materials * j]
            sig_sin[i][j] = data.iloc[10, 1 + i + unique_materials * j]
            sig_sout[i][j] = data.iloc[11, 1 + i + unique_materials * j]
            sig_f[i][j] = data.iloc[12, 1 + i + unique_materials * j]
            nu[i][j] = data.iloc[13, 1 + i + unique_materials * j]
            chi[i][j] = data.iloc[14, 1 + i + unique_materials * j]

    # Now we build a material map for each assembly.
    for i in xrange(0, assembly_types):  # loop over assembly types
        for j in xrange(0, int(assembly_cells / key_length[0][i])):  # loop over number of key lengths in each assembly type
            for k in xrange(0, int(key_length[0][i])):  # loop over key lengths
                local_map[i][j * int(key_length[0][i]) + k] = data.iloc[21 + 4 * i, k + 1]

    for i in xrange(0, assemblies):
        assembly_map[0][i] = data.iloc[17, i + 1]

    # The local assembly maps are then used to make a global material map from the geometry described in the Assembly Map
    # entry of the csv.
    material = np.array(local_map[int(data.iloc[17, 1])-1][:])  # initialize as first assembly in geometry

    # This loops concatenates additional assemblies to the global map as specified in the Assembly Map entry.
    for i in xrange(1, assemblies):
        material = np.concatenate((material,local_map[int(data.iloc[17, i + 1]) - 1][:]))

    # Subtract one from each material value to reflect the correct index, then convert array to integers.
    material = material - np.ones([1, cells])
    material = material.astype(int)

    print "File loaded."

    # Return relevant parameters.
    return sig_t, sig_sin, sig_sout, sig_f, nu, chi, groups, cells, cell_size, assembly_map.astype(int), material, assembly_size, assembly_cells

if __name__ == "__main__":
    print read_csv("AI_test.csv")