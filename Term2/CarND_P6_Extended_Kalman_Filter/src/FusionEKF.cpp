#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
	is_initialized_ = false;

	previous_timestamp_ = 0;

	// initializing matrices
	R_laser_ = MatrixXd(2, 2);
	R_radar_ = MatrixXd(3, 3);
	H_laser_ = MatrixXd(2, 4);
	H_jacobian = MatrixXd(3, 4);

	//measurement covariance matrix - laser
	R_laser_ << 0.0225, 0,
                0, 0.0225;

	//measurement covariance matrix - radar
	R_radar_ << 0.09, 0, 0,
				0, 0.0009, 0,
				0, 0, 0.09;

	// measurement matrix.
	H_laser_ << 1, 0, 0, 0,
				0, 1, 0, 0;

	// Acceleration noise componenets; Suggested values.
	noise_ax = 9;
	noise_ay = 9;

	// Initialize the kalman filter variables
	ekf_.P_ = MatrixXd(4, 4);
	ekf_.P_ << 1, 0, 0, 0,
			   0, 1, 0, 0,
			   0, 0, 1000, 0,
               0, 0, 0, 1000;

	ekf_.F_ = MatrixXd(4, 4);
	ekf_.F_ << 1, 0, 1, 0,
               0, 1, 0, 1,
               0, 0, 1, 0,
               0, 0, 0, 1;

	ekf_.Q_ = MatrixXd(4, 4);
	ekf_.x_ = VectorXd(4);
} // Constructor.
/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
	if (!is_initialized_) {
		VectorXd new_x = VectorXd(4);
		if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
			// convert polar to cartesian cordinates for radar.
			new_x = tools.PolarToCartesian(measurement_pack);
        }
        else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            new_x << measurement_pack.raw_measurements_[0],\
					measurement_pack.raw_measurements_[1], 0.0, 0.0;
        }

        // Address very low initial values.
	    if (fabs(new_x(0)) <  tools.EPS and fabs(new_x(1)) < tools.EPS) {
		    new_x(0) = tools.EPS;
		    new_x(1) = tools.EPS;
	    }
	    // first measurement
	    cout << "EKF: " << endl;
		ekf_.x_ = new_x;

	    // done initializing, no need to predict or update
	    previous_timestamp_ = measurement_pack.timestamp_;
	    is_initialized_ = true;
	    return;
	}  // Initilization.

   /*****************************************************************************
    *  Prediction
    ****************************************************************************/
   /**
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

	// Milliseconds to seconds.
	float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
	previous_timestamp_ = measurement_pack.timestamp_;

	ekf_.UpdateF(dt);
	ekf_.UpdateQ(dt, noise_ax, noise_ay);
	ekf_.Predict();

   /*****************************************************************************
   *  Update
   ****************************************************************************/
   /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
    */

	if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
		// Radar updates
		H_jacobian = tools.CalculateJacobian(ekf_.x_);
		ekf_.H_ = H_jacobian;
		ekf_.R_ = R_radar_;
		ekf_.UpdateEKF(measurement_pack.raw_measurements_);
	} else {
		// Laser updates
		ekf_.H_ = H_laser_;
		ekf_.R_ = R_laser_;
		ekf_.Update(measurement_pack.raw_measurements_);
   }

   // print the output
   // cout << "x_ = " << ekf_.x_ << endl;
   // cout << "P_ = " << ekf_.P_ << endl;
}
