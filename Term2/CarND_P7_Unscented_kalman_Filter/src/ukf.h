#ifndef UKF_H
#define UKF_H

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "measurement_package.h"
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
    public:

        ///* Stream to store NIS Lidar values.
        std::ofstream NIS_LidarStream_;

        ///* Stream to store NIS Radar values..
        std::ofstream NIS_RadarStream_;

        ///* initially set to false, true in first call of ProcessMeasurement
        bool is_initialized_;

        ///* Laser measurements will be ignored on false (except for init)
        bool use_laser_;

        ///* Radar measurements will be ignored on False(except for init)
        bool use_radar_;

        ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units
        VectorXd x_;

        ///* state covariance matrix
        MatrixXd P_;

        ///* Measurement matix.
        MatrixXd H_;

        ///* predicted sigma points matrix
        MatrixXd Xsig_pred_;

        ///* Process noise standard deviation longitudinal acceleration in m/s^2
        double std_a_;

        ///* Process noise standard deviation yaw acceleration in rad/s^2
        double std_yawdd_;

        ///* Vector for sigma points.
        VectorXd weights_;

        ///* Matrix to hold radar noise.
        MatrixXd R_radar_;

        ///* Matrix to hold laser noise.
        MatrixXd R_laser_;

        ///* State dimension
        int n_x_;

        ///* Agumented state dimension.
        int n_aug_;

        ///* Sigma agumented dimension.
        int n_a_sig_;

        ///* Sigma points spreading parameter.
        double lambda_;

        ///* current NIS for radar.
        double NIS_radar_;

        ///* Current NIS for laser.
        double NIS_laser_;

        ///* Vendor noise measurements.
        ///* Laser measurement noise standard deviation position1 in m
        double std_laspx_;

        ///* Laser measurement noise standard deviation position2 in m
        double std_laspy_;

        ///* Radar measurement noise standard deviation radius in m
        double std_radr_;

        ///* Radar measurement noise standard deviation angle in rad
        double std_radphi_;

        ///* Radar measurement noise standard deviation radius change in m/s
        double std_radrd_ ;

        ///* Previous timestamp
        long long previous_timestamp_;

        /**
         * Constructor
         */
        UKF();

        /**
         * Destructor
         */
        virtual ~UKF();

        /**
         * ProcessMeasurement
         * @param meas_package The latest measurement data of either radar or
         * laser
         */
        void ProcessMeasurement(MeasurementPackage meas_package);

        /**
         * Prediction Predicts sigma points, the state, and the state
         * covariance matrix
         * @param delta_t Time between k and k+1 in s
         */
        void Prediction(double delta_t);

        /**
         * Updates the state and the state covariance matrix using
         * laser measurement
         * @param meas_package The measurement at k+1
         */
        void UpdateLidar(MeasurementPackage meas_package);

        /**
         * Updates the state and the state covariance matrix using
         * radar measurement
         * @param meas_package The measurement at k+1
         */
        void UpdateRadar(MeasurementPackage meas_package);

        /**
         * Calculate Agumented sigma points.
         * @param MatrixXd* Agumented sigma points.
         */
        void GenerateSigmaPoints(MatrixXd &Xsig_out);

        /**
         * Predict Sigma points.
         * @param MatrixXd* Generated sigma points.
         * @param double time difference.
         */
        void PredictSigmaPoints(MatrixXd &Xsig_out, const double delta_t);

private:
        Tools tools;

};

#endif /* UKF_H */
