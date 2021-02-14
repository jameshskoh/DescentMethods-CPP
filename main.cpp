#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include "descentmethods.h"

double F(Eigen::VectorXd x) {
    return pow(x(0), 4.0) + pow(x(0), 2.0) * x(1) + 0.5 * pow(x(0), 2.0) + 3 * x(0) * x(1) + pow(x(1), 2.0);
}

Eigen::VectorXd J(Eigen::VectorXd x) {
    int x_size = x.size();

    Eigen::VectorXd jacobian(x_size);

    jacobian(0) = 4 * pow(x(0), 3.0) + 2 * x(0) * x(1) + x(0) + 3 * x(1);
    jacobian(1) = pow(x(0), 2.0) + 3 * x(0) + 2 * x(1);

    return jacobian;
}

Eigen::MatrixXd H(Eigen::VectorXd x) {
    int x_size = x.size();

    Eigen::MatrixXd hessian(x_size, x_size);

    hessian(0, 0) = 12 * pow(x(0), 2.0) + 2 * x(1) + 1;
    hessian(0, 1) = 2 * x(0) + 3;
    hessian(1, 0) = 2 * x(0) + 3;
    hessian(1, 1) = 2;

    return hessian;
}

int main() {
    Eigen::Vector2d x0;

    x0(0) = -0.5;
    x0(1) = -0.5;

    double epsilon = 0.00001;
    double alpha = 0.3;
    double beta = 0.5;

    Eigen::VectorXd x = DM::Newton(x0, epsilon, alpha, beta, F, J, H);

    std::cout << "The solution is: " << x << std::endl;
    std::cout << "The objective is: " << F(x) << std::endl;
}
