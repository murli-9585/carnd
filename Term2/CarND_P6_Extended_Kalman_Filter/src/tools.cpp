#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size

	if (estimations.size() != ground_truth.size()) {
		std::cout << "Estimation not equal to ground truth!" << std::endl;
		return rmse;
	}
	if (estimations.size() == 0){
		std::cout << "Invalid estimation" << std::endl;
		return rmse;
	}

	//accumulate squared residuals
	for(unsigned int i=0; i < estimations.size(); ++i){

		VectorXd residual = estimations[i] - ground_truth[i];

		//coefficient-wise multiplication
		residual = residual.array()*residual.array();
		rmse += residual;
	}

	//calculate the mean
	rmse = rmse/estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	MatrixXd Hj(3,4);
	Hj << 0,0,0,0,
          0,0,0,0,
          0,0,0,0;

	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//pre-compute a set of terms to avoid repeated calculation
	float c1 = px*px+py*py;
	float c2 = sqrt(c1);
	float c3 = (c1*c2);

	//check division by zero
	if(fabs(c1) < EPS){
	std::cout << "Function CalculateJacobian(): Correcting Division by zero."\
			 << std::endl;
	c1 = EPS;
	}

  //compute Jacobian matrix
  Hj << (px/c2), (py/c2), 0, 0,
        -(py/c1), (px/c1), 0, 0,
        py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj;
}

VectorXd Tools::PolarToCartesian(const MeasurementPackage &measurement_pack) {

	float px, py, vx, vy;
	// range.
	float rho = measurement_pack.raw_measurements_[0];
	float phi = measurement_pack.raw_measurements_[1];
    // velocity.
	float rho_dot = measurement_pack.raw_measurements_[2];

	px = rho * cos(phi);
	py = rho * sin(phi);
	vx = rho_dot * cos(phi);
	vy = rho_dot * sin(phi);
	VectorXd new_x = VectorXd(4);
	new_x << px, py, vx, vy;
	return new_x;
}

VectorXd Tools::CartesianToPolar(const VectorXd &x_state) {

	float px, py, vx, vy;
	px = x_state[0];
	py = x_state[1];
	vx = x_state[2];
	vy = x_state[3];

	float rho, phi, rho_dot;
	rho = sqrt(px*px + py*py);
	// if rho is very small, set it to 0.000001 to avoid division by 0.
	if (fabs(rho) < EPS)
		rho = EPS;

	phi = atan2(py, px);  // returns values between -pi and pi
	rho_dot = (px * vx + py * vy) / rho;

	VectorXd z_pred = VectorXd(3);
	z_pred << rho, phi, rho_dot;

	return z_pred;
}
