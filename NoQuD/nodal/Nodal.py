
import numpy as np

class Nodal:

    def __init__(self, diffusion_constant, sigma_r, cell_size, f, groups, n):

        repeated_coefficients = np.array([[0, 1, -3, 6, -10],
                                 [0, 1, 3, 6, 10],
                                 [0, 0, -12, 0, -40],
                                 [0, 0, 0, -60, 0],
                                 [0, 0, 0, 0, -140],
                                 [0, 1, -3, 6, -10]])

        print repeated_coefficients




if __name__ == "__main__":

    test = Nodal(1, 1, 1, 1, 1, 1)
