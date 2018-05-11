#include "PID.h"
/*
 * PID (Proportional, Integral, Differential) Control class.
 *
 */


PID::PID() {}

PID::~PID() {
}

void PID::Init(double Kp, double Ki, double Kd, bool use_twiddle, int n_eval) {
        this->p_error = 0.0;
        this->d_error = 0.0;
        this->i_error = 0.0;

        // Initialize
        this->Kp = Kp;
        this->Kd = Kd;
        this->Ki = Ki;

        this->eval_state = start;
        this->pid_param = proportional;

        // Initial values of evaluvating params.
        // Should be factor of params??
        // XXXMM Factor of params seems more effective.
        this->dp[0] = 0.1*Kp;
        this->dp[1] = 0.1*Kd;
        this->dp[2] = 0.1*Ki;

        // Parameter optimization initializations.
        this->optimize_params = use_twiddle;
        this->n_steps = n_eval; // start evaluating after n steps.
        this->step_i = 1;
        this->best_err = std::numeric_limits<double>::max();
        this->total_err = 0.0;
}

void PID::UpdateError(double cte) {
        previous_cte = p_error;
        p_error = cte; // Directly proportional to error
        i_error = i_error + cte; // Integral of all errors.
        d_error = cte - previous_cte; // Differencial
}

double PID::TotalError() {

        // Update total error only after passing inital noisy steps.
        if (optimize_params &&
            (step_i % (n_steps + noisy_steps) > noisy_steps))
                total_err += pow(p_error, 2);

        // Step 1: Find optimal parameters.
        if (optimize_params &&
            (step_i % (n_steps + noisy_steps) == 0) &&
            ((dp[0]+dp[1]+dp[2]) > THRESHOLD)) {
                // For every evaluation, re-set total error.
                double  p_value = UpdateParameters();
                switch (pid_param) {
                case proportional:
                        Kp += p_value;
                        break;
                case differential:
                        Kd += p_value;
                        break;
                case integral:
                        Ki += p_value;
                        break;
                default:
                        break;
                };
                // DEBUG
                std::cout << "--------------------" << std::endl;
                std::cout << "OPTIMIZED PARAMS" << std::endl;
                std::cout << " P: " << Kp << " D: " << Kd << " I: " << Ki \
                    << std::endl;
                std::cout << " dp_p: " << dp[0] << " dp_d: " << dp[1] << " dp_i: "\
                    << dp[2] << std::endl;
                std::cout << " Total Error: " << total_err << std::endl;
                std::cout << " CTE: " << p_error << std::endl;
                std::cout << "--------------------" << std::endl;
                // reset total error.
                total_err = 0.0;
        }
        step_i++;
        // Step 2: Calculate Total error.
        return -Kp*p_error - Ki*i_error - Kd*d_error;
}

double PID::UpdateParameters() {
        // Implementation of Twiddle to optimize control parameters.
        double temp_c = 0.0;
        switch (eval_state) {
        case start:
                temp_c = dp[pid_param];
                eval_state = positive_eval;
                break;
        case positive_eval:
                if (total_err < best_err) {
                        // evaluation experiment is correct.
                        dp[pid_param] *= 1.1;
                        // re-set eval_state, update parameter state
                        // and return.
                        pid_param = static_cast<PIDParameters>(
                            (pid_param+1) % 3);
                        eval_state = start;
                        temp_c = dp[pid_param];
                        best_err = total_err;
                }
                else {
                        // Decrement the value and re-try.
                        temp_c = -2.0*dp[pid_param];
                        eval_state = negative_eval;
                }
                break;
        case negative_eval:
                if (total_err < best_err) {
                        // Evaluation is right, rotate param.
                        dp[pid_param] *= 1.1;
                        best_err = total_err;
                }
                else {
                        dp[pid_param] *= 0.9;
                }
                // Move to evaluate next param and reset state.
                pid_param = static_cast<PIDParameters>(
                    (pid_param+1) % 3);
                temp_c = dp[pid_param];
                eval_state = start;
                break;
        default:
                break;
        };
        return temp_c;
}

