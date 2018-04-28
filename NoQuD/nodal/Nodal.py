
import numpy as np

class Nodal:

    def __init__(self, diffusion_constant, sigma_r, cell_size, f, groups, nodes):

        self.repeated_coefficients = np.array([[0, 1, -3, 6, -10],
                                 [0, 1, 3, 6, 10],
                                 [0, 0, -12, 0, -40],
                                 [0, 0, 0, -60, 0],
                                 [0, 0, 0, 0, -140],
                                 [0, 1, -3, 6, -10]])
        self.diffusion_constant = diffusion_constant
        self.sigma_r = sigma_r
        self.cell_size = cell_size
        self.f = f
        self.groups = groups
        self.nodes = nodes
        self.order_of_legendre_poly = 5
        self.linear_system = np.zeros((self.groups, self.order_of_legendre_poly * self.nodes,
                                       self.order_of_legendre_poly * self.nodes))
        self.lhs_boundary_condition()
        self.rhs_boundary_condition()

    def lhs_boundary_condition(self):

        for i in xrange(0, self.groups):
            for j in xrange(0, self.order_of_legendre_poly):

                self.linear_system[i][0][j] = self.repeated_coefficients[0][j]

    def rhs_boundary_condition(self):

        for i in xrange(0, self.groups):
            for j in xrange(0, self.order_of_legendre_poly):

                self.linear_system[i][1][-j] = self.repeated_coefficients[1][-j]




if __name__ == "__main__":

    test = Nodal(1, 1, 1, 1, 2, 2)
    print test.linear_system
