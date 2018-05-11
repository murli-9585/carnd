#ifndef PID_H
#define PID_H

#include <cmath>
#include <iostream>
#include <stdlib.h>
/*
 * Track which co-efficient is being evaluated.
 */
enum EvaluationState {
        start,
        positive_eval,
        negative_eval,
        last=negative_eval
};

/*
 * Enumarator to keep track to parameter under evaluvation.
 */
enum PIDParameters {
        proportional=0,
        differential=1, // When using Index, PDI execution is followed
        integral=2,
};

/* PID Class: Proportional, Integral and Differential terms)
 * which help calculate correct value of a control. Control
 * can be Velocity, steering, throttle etc.,
 * P: Term p for Proportionality for the error noticed from
 * prior reading. For ex: If the robot/car is moving in X-Axis
 * error is proportional to how y-value is maintained. That difference
 * in y is error. P is the proportinal factor of that error.
 * D: Differencial factor which helps in overshooting correction from
 * other factors (could be P or I factors as well). This factor helps in
 * determining the oscillation of car, the better factor the car does not
 * osscillate.
 * I: Integral Factor; Its a summation of all errors from previous readings
 * Sustained measure of all errors to avoid moving closer or overshooting.
 */
class PID {
    public:

        /*
         * Track evaluation state of parameter.
         */
        EvaluationState eval_state;

        /*
         * Track parameter under evaluation.
         */
        PIDParameters pid_param;

        /*
         * Errors
         */
        double p_error;
        double d_error;
        double i_error;

        /*
         * Coefficients
         */
        double Kp;
        double Kd;
        double Ki;

        /*
         * Coefficient factors for P, D  & I.
         */
        double dp[3];

        /*
         * Previous CTE.
         */
        double previous_cte;

        /*
         * Threshold error rate.
         */
        const double THRESHOLD = 0.001;

        /*
         * Optimize parameters (P, I, D). Uses twiddle.
         */
        bool optimize_params;

        /*
         * Number of steps to evaluvate to optimize a param.
         */
        int n_steps;

        /*
         * Current step; Step is determined by the simulator.
         */
        int step_i;

        /*
         * Initial noisy steps due to change in parameter.
         */
        const int noisy_steps = 50;

        /*
         * Current best error noted by evaluvation.
         */
        double best_err;

        /*
         * Tracking Total error.
         */
        double total_err;

        /*
         * Constructor
         */
        PID();

        /*
         * Destructor.
         */
        virtual ~PID();

        /*
         * Initialize PID.
         */
        void Init(double Kp, double Ki, double Kd,
                  bool use_twiddle, int n_eval);

        /*
         * Update the PID error variables given cross track error.
         */
        void UpdateError(double cte);

        /*
         * Calculate the total PID error.
         */
        double TotalError();

        /*
         * Method evaluvates parameters using Twiddle algorithm.
         */
        double UpdateParameters();
};

#endif /* PID_H */
