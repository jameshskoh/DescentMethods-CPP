//
// Created by James Koh on 03/02/2021.
//

#ifndef DESCENTMETHODS_DESCENTMETHODS_H
#define DESCENTMETHODS_DESCENTMETHODS_H

#include <iostream>
#include <Eigen/Dense>

#endif //DESCENTMETHODS_DESCENTMETHODS_H

namespace DM {
    using namespace Eigen;

    MatrixXd Direction(VectorXd jacobian, MatrixXd hessian_inverse);

    double LambdaSquared(VectorXd jacobian, MatrixXd hessian_inverse);

    VectorXd LineSearchUpdate(VectorXd x, VectorXd delta_x, double alpha, double beta,
                              double F (VectorXd), VectorXd jacobian);

    VectorXd Newton(VectorXd x0, double epsilon, double alpha, double beta,
                    double F (VectorXd), VectorXd J (VectorXd), MatrixXd H (VectorXd));
}