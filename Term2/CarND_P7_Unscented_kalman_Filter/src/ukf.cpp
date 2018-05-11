#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
        // Use laser measurements.
        use_laser_ = true;

        // Use radar measurements on true.
        use_radar_ = true;

        // initial state vector
        x_ = VectorXd(5);

        // initial covariance matrix
        P_ = MatrixXd(5, 5);
        P_ << 1, 0, 0, 0, 0,
           0, 1, 0, 0, 0,
           0, 0, 1, 0, 0,
           0, 0, 0, 1, 0,
           0, 0, 0, 0, 1;


        // Process noise standard deviation longitudinal acceleration in m/s^2
        std_a_ = 0.5;

        // Process noise standard deviation yaw acceleration in rad/s^2
        std_yawdd_ = 0.5;

        // First measurement.
        is_initialized_ = false;

        // State dimension.
        n_x_ = 5;

        // State agument dimension.
        n_aug_ = 7;

        // Sigma augument dimension.
        n_a_sig_ = 2 * n_aug_ + 1;

        // spreading param.
        lambda_ = 3 - n_aug_;

        // Sigma points matrix.
        Xsig_pred_ = MatrixXd(n_x_, n_a_sig_);

        // Weights of sigma points.
        weights_ = VectorXd(n_a_sig_);

        // Update weights.
        weights_(0) = lambda_ / (lambda_+n_aug_);
        for (int i=1; i<n_a_sig_; i++)
                weights_(i) = 0.5 / (lambda_+n_aug_);

        // Measurement matrix
        H_ = MatrixXd(2, n_x_);
        H_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;

        // Noise matrix for ladar and Radar.
        R_radar_ = MatrixXd(3, 3);
        R_laser_ = MatrixXd(2, 2);

        NIS_radar_ = 0;
        NIS_laser_ = 0;

        previous_timestamp_ = 0.0;

        // Following values are noise approximation for ladar/radar
        // manufacturers.
        // Laser measurement noise standard deviation position1 in m
        std_laspx_ = 0.15;

        // Laser measurement noise standard deviation position2 in m
        std_laspy_ = 0.15;

        // Radar measurement noise standard deviation radius in m
        std_radr_ = 0.3;

        // Radar measurement noise standard deviation angle in rad
        std_radphi_ = 0.03;

        // Radar measurement noise standard deviation radius change in m/s
        std_radrd_ = 0.3;

        // Update radar noise matrix.
        R_radar_ << pow(std_radr_, 2), 0, 0,
                0, pow(std_radphi_, 2), 0,
                0, 0, pow(std_radrd_, 2);

        // Udate laser noise matrix.
        R_laser_ << pow(std_laspx_, 2), 0,
                 0, pow(std_laspy_, 2);

        // NIS Ladar data.
        NIS_LidarStream_.open( "../output/NIS/lidar.txt", ios::out );

        // NIS Radar data.
        NIS_RadarStream_.open( "../output/NIS/radar.txt", ios::out );
}

UKF::~UKF() {
        // Close the streams.
        NIS_LidarStream_.close();
        NIS_RadarStream_.close();
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

        // cout << "UKF" << endl;
        // First measurement.
        if (!is_initialized_) {
                if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
                        double rho = meas_package.raw_measurements_[0];
                        double phi = meas_package.raw_measurements_[1];
                        x_ << rho * cos(phi), rho * sin(phi), 0, 0, 0;
                }
                else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
                        x_ <<  meas_package.raw_measurements_[0],\
                            meas_package.raw_measurements_[1], 0, 0, 0;

        // XXXMM: May be update with constant for near zero values?

        is_initialized_ = true;
        previous_timestamp_ = meas_package.timestamp_;
        } // Initialized.

        double dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
        previous_timestamp_ = meas_package.timestamp_;

        // Predict with deltaT; dt.
        Prediction(dt);

        // Update step.
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
                UpdateRadar(meas_package);
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
                UpdateLidar(meas_package);
        }

} // ProcessMeasurement.

/**
 * Predicts sigma points, modify state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

        // Sigma points.
        MatrixXd Xsig_aug = MatrixXd(n_aug_, n_a_sig_);

        // Step1: Get Augumented sigma points based on x and augumented
        // mean and covarience.
        GenerateSigmaPoints(Xsig_aug);

        //Step2: Predict sigma points;Update Xsig_pred_
        PredictSigmaPoints(Xsig_aug, delta_t);

        //Step3: Predict mean and co-varience matrix.
        // Reset and Predict state mean.
        x_.fill(0.0);
        for (int i=0; i<n_a_sig_; i++)
                x_ = x_ + weights_(i) * Xsig_pred_.col(i);

        // Reset and Predict state covariance matrix.
        P_.fill(0.0);
        for (int i=0; i<n_a_sig_; i++) {
                // Iterate over sigma points; Normalize angle and update P_.
                VectorXd x_diff = Xsig_pred_.col(i) - x_;
                // Normalize.
                x_diff(3) = tools.Normalize(x_diff(3));
                //while (x_diff(3) > M_PI) x_diff(3) -= 2.0 * M_PI;
                //while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;

                P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
        }
}

/**
 * Calculate augumented sigma points.
 * @param {MatrixXd} Augumented sigma points to predict.
 */

void UKF::GenerateSigmaPoints(MatrixXd &Xsig_aug) {

        // Augmented mean.
        VectorXd x_aug = VectorXd(n_aug_);

        // Augumented state covarience.
        MatrixXd P_aug  = MatrixXd(n_aug_, n_aug_);

        // Fill augumented mean state.
        // Augument matrix has 2 more elements (Va, Vrodot) than x_.
        x_aug.head(n_x_) = x_;
        x_aug(n_aug_-2) = 0;
        x_aug(n_aug_-1) = 0;

        // Fill augumented covarience matrix.
        P_aug.fill(0.0);
        P_aug.topLeftCorner(n_x_, n_x_) = P_;
        P_aug(n_x_, n_x_) = std_a_ * std_a_;
        P_aug(n_x_+1, n_x_+1) = std_yawdd_ * std_yawdd_;

        // Squareroot matrix.
        MatrixXd L = P_aug.llt().matrixL();

        // Create augumented sigma points.
        Xsig_aug.col(0) = x_aug;
        for (int i=0; i<n_aug_; i++) {
                Xsig_aug.col(i+1) = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
                Xsig_aug.col(i+1+n_aug_) = \
                    x_aug - sqrt(lambda_+n_aug_) * L.col(i);
        }
}

/**
 * Predict sigma points.
 * @param {MatriXd} Generated sigma points.
 */

void UKF::PredictSigmaPoints(MatrixXd &Xsig_aug, const double delta_t) {
        for (int i=0; i<n_a_sig_; i++) {
                double p_x = Xsig_aug(0, i);
                double p_y = Xsig_aug(1, i);
                double v = Xsig_aug(2, i);
                double yaw = Xsig_aug(3, i);
                double yawd = Xsig_aug(4, i);
                double nu_a = Xsig_aug(5, i);
                double nu_yawdd = Xsig_aug(6, i);

                // state values.
                double px_p, py_p;

                // Division by zero escape.
                if (fabs(yawd) > 0.001) {
                        px_p = p_x + v/yawd * (sin (yaw+yawd*delta_t)-sin(yaw));
                        py_p = p_y + v/yawd * (cos(yaw)-cos(yaw+yawd*delta_t));
                }
                else {
                        px_p = p_x + v * delta_t * cos(yaw);
                        py_p = p_y + v * delta_t * sin(yaw);
                }

                double v_p = v;
                double yaw_p = yaw + yawd * delta_t;
                double yawd_p = yawd;

                // Add noise.
                px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
                py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
                v_p  = v_p + nu_a * delta_t;

               yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
               yawd_p = yawd_p + nu_yawdd * delta_t;

               // Update predicted sigma points.
               Xsig_pred_(0, i) = px_p;
               Xsig_pred_(1, i) = py_p;
               Xsig_pred_(2, i) = v_p;
               Xsig_pred_(3, i) = yaw_p;
               Xsig_pred_(4, i) = yawd_p;
       }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

        // Use lidar data to update the belief about the object's
        // position. Modify the state vector, x_, and covariance, P_.
        VectorXd z_pred = H_ * x_;

        VectorXd z = meas_package.raw_measurements_;
        VectorXd y = z - z_pred;
        MatrixXd Ht = H_.transpose();
        MatrixXd S = H_ * P_ * Ht + R_laser_;
        MatrixXd Si = S.inverse();
        MatrixXd PHt = P_ * Ht;
        // Kalman gain.
        MatrixXd K = PHt * Si;

        x_ = x_ + (K * y);
        long size_x = x_.size();
        MatrixXd I = MatrixXd::Identity(size_x, size_x);
        P_ = (I - K * H_) * P_;

        // NIS laser.
        NIS_laser_ = y.transpose() * S.inverse() * y;
        NIS_LidarStream_ << NIS_laser_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
        // Use radar data to update the belief about the object's
        // position. Modify the state vector, x_, and covariance, P_.

		// Radar has 3 dimensions, x, phi, and r_dot.
        int n_z = 3;
        MatrixXd Zsig = MatrixXd(n_z, n_a_sig_);

        // Transform sigma points to measurement space.
        for (int i=0; i<Xsig_pred_.cols(); i++) {
                double p_x = Xsig_pred_(0,i);
                double p_y = Xsig_pred_(1,i);
                double v = Xsig_pred_(2,i);
                double yaw = Xsig_pred_(3,i);

                double v1 = cos(yaw) * v;
                double v2 = sin(yaw) * v;

                double dist = sqrt(p_x*p_x + p_y*p_y);
                Zsig(0,i) = dist; // r
                Zsig(1,i) = atan2(p_y, p_x); // phi
                Zsig(2, i) = (p_x*v1 + p_y*v2) / dist; //r_dot

        }

        // Calculate mean z_pred and covariance S of predicted points.
        VectorXd z_pred = VectorXd(n_z);
        z_pred.fill(0.0);

        for (int i=0; i<Zsig.cols(); i++)
                z_pred += (weights_(i) * Zsig.col(i));

        // Measurement covariance matrix S.
        MatrixXd S = MatrixXd(n_z, n_z);
        S = R_radar_;
        for (int i=0; i<Zsig.cols(); i++) {
                VectorXd diff = Zsig.col(i) - z_pred;
                diff(1) = tools.Normalize(diff(1));
                //while (diff(1) > M_PI) diff(1) -= 2.0*M_PI;
                //while (diff(1) < -M_PI) diff(1) += 2.0*M_PI;
                S += (weights_(i) * (diff * diff.transpose()));
        }

        // Calculate cross-correleation matix.
        MatrixXd Tc = MatrixXd(n_x_, n_z);
        Tc.fill(0.0);
        for (int i=0; i< n_a_sig_; i++) {
                VectorXd diff_x = (Xsig_pred_.col(i) - x_);
                diff_x(3) = tools.Normalize(diff_x(3));
                //while (diff_x(3) > M_PI) diff_x(3) -= 2.0*M_PI;
                //while (diff_x(3) < -M_PI) diff_x(3) += 2.0*M_PI;

                VectorXd diff_z = Zsig.col(i) - z_pred;
                diff_z(1) = tools.Normalize(diff_z(1));
                //while (diff_z(1) > M_PI) diff_z(1) -= 2.0*M_PI;
                //while (diff_z(1) < -M_PI) diff_z(1) += 2.0*M_PI;

                Tc += (weights_(i) * (diff_x * diff_z.transpose()));
        }

        MatrixXd K(n_x_, n_z);
        K = Tc * S.inverse();

        // Residual
        VectorXd z = meas_package.raw_measurements_;
        VectorXd z_diff = z - z_pred;

        // angle normalization.
        z_diff(1) = tools.Normalize(z_diff(1));
        // while (z_diff(1) > M_PI) z_diff(1) -= 2.0*M_PI;
        // while (z_diff(1) < -M_PI) z_diff(1) += 2.0*M_PI;

        // update state mean and covariance matrix
        x_ = x_ + K * z_diff;
        P_ = P_ - K * S * K.transpose();

        NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
        NIS_RadarStream_ << NIS_radar_ << endl;
}
