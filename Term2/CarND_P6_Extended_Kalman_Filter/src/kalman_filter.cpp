#include <math.h>
#include "kalman_filter.h"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
	// Predict the state.
	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	VectorXd z_pred = VectorXd(3);
	z_pred = H_ * x_;
	VectorXd y = z - z_pred;

	CalculateNewEstimate(z, y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
	VectorXd z_pred = tools.CartesianToPolar(x_);
	VectorXd y = z - z_pred;
	// Normalize angle between -PI and PI.
	while (y(1) > M_PI) {
		y(1) -= 2 * M_PI;
	}
	while (y(1) < -M_PI) {
		y(1) += 2 * M_PI;
	}
	CalculateNewEstimate(z, y);
}

void KalmanFilter::CalculateNewEstimate(const VectorXd &z,
		const VectorXd &y) {

	MatrixXd Ht = H_.transpose();
	MatrixXd PHt = P_ * Ht;
	MatrixXd S = H_ * PHt + R_;
	MatrixXd Si = S.inverse();
	MatrixXd K = PHt * Si;

	// New estimates for x and P
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::Updatex(double px, double py, double vx, double vy) {
    x_ << px, py, vx, vy;
}

void KalmanFilter::UpdateF(float dt) {
    F_(0, 2) = dt;
	F_(1, 3) = dt;
}

void KalmanFilter::UpdateQ(float dt, float noise_ax, float noise_ay) {
	float dt2 = dt * dt;
	float dt3 = dt2 * dt;
	float dt4 = dt3 * dt;

	Q_ << dt4/4*noise_ax, 0, dt3/2*noise_ax, 0,
		  0, dt4/4*noise_ay, 0, dt3/2*noise_ay,
		  dt3/2*noise_ax, 0, dt2*noise_ax, 0,
		  0, dt3/2*noise_ay, 0, dt2*noise_ay;
}

