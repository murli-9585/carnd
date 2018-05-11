#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"
#include "measurement_package.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Tools {
public:
   /**
   * Constructor.
   */
   Tools();

   /**
   * Destructor.
   */
   virtual ~Tools();

   /**
   * A helper method to calculate RMSE.
   */
   VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
            const vector<VectorXd> &ground_truth);

   /**
   * A helper method to calculate Jacobians.
   */
   MatrixXd CalculateJacobian(const VectorXd& x_state);

   /**
   * Convert polar to cartesian measurements.
   */
    VectorXd PolarToCartesian(const MeasurementPackage &measurement_pack);

    /**
     * Convert cartesian to polar measurement.
     */
    VectorXd CartesianToPolar(const VectorXd &x_state);

    /**
     * EPS used in caculating divison by zero.
     */
      const float EPS = 0.000001;
};

#endif /* TOOLS_H_ */
