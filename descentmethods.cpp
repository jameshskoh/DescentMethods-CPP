//
// Created by James Koh on 03/02/2021.
//

#include <iostream>
#include <Eigen/Dense>

namespace DM {
    using namespace Eigen;

    MatrixXd Direction(VectorXd jacobian, MatrixXd hessian_inverse) {
        return - hessian_inverse * jacobian;
    }

    double LambdaSquared(VectorXd jacobian, MatrixXd hessian_inverse) {
        return jacobian.transpose() * hessian_inverse * jacobian;
    }

    VectorXd LineSearchUpdate(VectorXd x, VectorXd delta_x, double alpha, double beta,
                              double F (VectorXd), VectorXd jacobian) {
        double t = 1.0;

        while (F(x + t * delta_x) > (F(x) + jacobian.dot(delta_x) * (alpha * t))) {
            t = beta * t;
        }

        return x + t * delta_x;
    }

    VectorXd Newton(VectorXd x0, double epsilon, double alpha, double beta,
                    double F (VectorXd), VectorXd J (VectorXd), MatrixXd H (VectorXd)) {
        VectorXd x = x0;
        int i = 1;

        while (true) {
            std::cout << "Iteration " << i << ":" << std::endl;

            VectorXd jacobian = J(x);
            MatrixXd hessian = H(x);
            MatrixXd hessian_inverse = H(x).inverse();

            VectorXd delta_x = Direction(jacobian, hessian_inverse);
            double lambda_squared = LambdaSquared(jacobian, hessian_inverse);

            std::cout << "Lambda^2: " << lambda_squared << std::endl;

            if (lambda_squared / 2 < epsilon || i > 100)
                break;

            x = LineSearchUpdate(x, delta_x, alpha, beta, F, jacobian);
            std::cout << "Solution :" << F(x) << std::endl;

            i++;
        }

        return x;
    }
}