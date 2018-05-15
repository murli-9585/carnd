#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// Set the timestep length and duration
size_t N = 8;
double dt = 0.08;
// Velocity for reference.
// Case 1: N=10, df=0.1 ref_v = 65 Sharp curves caused green line to deviate
// Tried with multiple Delta and accelerator values. From  100, 200, 300.
// Case 2: Implemented Sequential difference: this seems logical and referenced
// Automotive Model Predictive control by  Luigi Del ri.
// Case 3: Tried lot of factors 10, 20, 30, 50, 100 as multiplying factor between
// delta, accelerator to Factor * sequential difference. - Great imporvement in
// the vehicle being in-lanes and reduction in sudden spike in delta.
// Case 4: Reduced N to 8 and dt to 0.09 (This impacted CTE) and the vehicle
// started to drift away from lane lines and cte seems to be low.
// Case 5: Kept N at 8, dt to 0.08. Very promising result at speed 65.
// Case 6: speed 70-80: Differential control is affected. The vehicle start
// To drift a lot before going outside the lanes.
// The smaller the difference the more affect it has on prediction and responds
// to lot of noise. I think the multiplying factors impacted quite a bit.
// Having df close to 0.08 - 0.1 seems reasonable. Further improvement in
// finding effective cost might be needed to adjust to any speed.
// Note: For most of these cases, cost factors are tweaked a bit but I feel it
// has almost no affect.
// At ref_v 65, Vehicle was in middle of lane and at 75 its good speed but
// real test to throttle and steering angle. It did not crossed the track
// but touched the lane line only once. The green line did not wiggled at lot
// meaning its cte and epsi were close to zero.
double ref_v = 75;

// Lf is the distance between middle of first 2 wheels and middle of car.
// Lf is obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on
// a flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// Cost Factors for actuators.
double cost_delta = 200;
double cost_a = 0.5;
double seq_factor = 100;

// Solver takes all the start variables and actuator variables in a single
// vector. These values will help figuring out the index of each variable.
size_t x_start = 0; // X values will be in index 0 to N-1.
size_t y_start = x_start + N; // Y values from x_start+N to x_start+N+N-1
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N-1;

// fg start values differ from vars start values. as fg[0] is
// reserved for cost.
size_t fg_x_start = 1; // X values will be in index 1 to N.
size_t fg_y_start = fg_x_start + N; // Y values from x_start+N to x_start+N+N-1
size_t fg_psi_start = fg_y_start + N;
size_t fg_v_start = fg_psi_start + N;
size_t fg_cte_start = fg_v_start + N;
size_t fg_epsi_start = fg_cte_start + N;
size_t fg_delta_start = fg_epsi_start + N;
size_t fg_a_start = fg_delta_start + N-1;

class FG_eval {
public:
        // Fitted polynomial coefficients
        Eigen::VectorXd coeffs;
        FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

        typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
        void operator()(ADvector& fg, const ADvector& vars) {
                // fg: Vector for cost & constraints.
                // vars: Vector for state values(x,y, velocity, etc)
                //       & Actuators(steering angle and accleration)

                // Cost; fg[0] holds cost value.
                // Cost: How far the cte and orientation are from 0.
                // Its determined as affect of these distances from 0.
                // Where 0 means prediction is in-line. The affects of
                // each vars and actuators effect cost.

                // XXXMM: Since values are initialized to 0. We can
                // Reduce the loop using while vars[cte_start..+t] >0
                // Cost i): Based on reference state.
                fg[0] = 0;
                for (int t=0; t<N; ++t) {
                        fg[0] += 5*CppAD::pow(vars[cte_start + t], 2);
                        fg[0] += 5*CppAD::pow(vars[epsi_start + t], 2);
                        fg[0] += CppAD::pow(vars[v_start + t] - ref_v, 2);
                }

                // Cost ii): Actuators effect on cost.
                // Note N-1 as last 'df' is non-important
                for (int t=0; t<N-1; ++t) {
                        fg[0] += cost_delta * CppAD::pow(vars[delta_start + t], 2);
                        fg[0] += cost_a * CppAD::pow(vars[a_start + t], 2);
                }

                // Cost iii): Minimize the value gap between sequential
                // actuations. This plays a larger role so multiply with large
                // number. ex: 500.
                for (int t=0; t<N-2; ++t) {
                        fg[0] += (seq_factor * cost_delta) * CppAD::pow(
                                vars[delta_start + t + 1] -
                                vars[delta_start + t], 2);
                        fg[0] += (seq_factor * cost_a) * CppAD::pow(
                                vars[a_start + t + 1] -
                                vars[delta_start + t], 2);

                        // Use sequential CTE error and EPSI errors.
                        fg[0] +=  CppAD::pow(vars[cte_start + t + 1] -
                                vars[cte_start + t], 2);
                        fg[0] +=  CppAD::pow(vars[epsi_start + t + 1] -
                                vars[epsi_start + t], 2);
                }

                // Setup Constraints.

                // Initial constraints.
                fg[fg_x_start] = vars[x_start];
                fg[fg_y_start] = vars[y_start];
                fg[fg_psi_start] = vars[psi_start];
                fg[fg_v_start] = vars[v_start];
                fg[fg_cte_start] = vars[cte_start];
                fg[fg_epsi_start] = vars[epsi_start];

                // Values at time t_1; "time-minus-1"
                AD<double> xt_1, yt_1, psit_1, vt_1, ctet_1, \
                    epsit_1, deltat_1, at_1, fxt_1;
                // Values recorded at time t.
                AD<double> xt, yt, psit, vt, ctet, epsit;

                AD<double> psi_dest_1;
                // Evaluvate constraints at time T.
                for (int t=1; t<N; t++) {
                        // Initialize previous values.
                        xt_1 = vars[x_start + t - 1];
                        yt_1 = vars[y_start + t - 1];
                        psit_1 = vars[psi_start + t - 1];
                        vt_1 = vars[v_start + t - 1];
                        ctet_1 = vars[cte_start + t - 1];
                        epsit_1 = vars[epsi_start + t - 1];
                        deltat_1 = vars[delta_start + t - 1];
                        at_1 = vars[a_start + t - 1];


                        // F(x) at t-1; Since this is a 3rd order polynomial
                        // use as c0 +c1*x +c2*x^2 + c3^x3.
                        fxt_1 = coeffs[0] + coeffs[1] * xt_1 + \
                            coeffs[2] * pow(xt_1, 2) + coeffs[3] * pow(xt_1, 3);
                        psi_dest_1 = CppAD::atan(coeffs[1] + \
                            (2 * coeffs[2] * xt_1) + (3 * coeffs[3] * pow(xt_1, 2)));

                        // Current Values at t.
                        xt = vars[x_start + t];
                        yt = vars[y_start + t];
                        psit = vars[psi_start + t];
                        vt = vars[v_start + t];
                        ctet = vars[cte_start + t];
                        epsit = vars[epsi_start + t];

                        // Apply Kinematic formulas to calculate at time "t"
                        // xt = xt_1 + vt_1 * cos(psi) * dt.
                        // so, differene/change in x between time t and t-1 is
                        // noted.
                        fg[fg_x_start + t] = xt -
                            (xt_1 + vt_1 * CppAD::cos(psit_1) * dt);

                        fg[fg_y_start + t] = yt -
                            (yt_1 + vt_1 * CppAD::sin(psit_1) * dt);

                        fg[fg_psi_start + t] = psit - (
                            psit_1 + ((vt_1/Lf) * deltat_1 * dt));

                        fg[fg_v_start + t] = vt - (vt_1 + at_1 * dt);

                        // Errors
                        fg[fg_cte_start + t] = ctet - ((fxt_1 - yt_1) +
                            (vt_1 * CppAD::sin(epsit_1) * dt));

                        fg[fg_epsi_start + t] = epsit - ((psit_1 - psi_dest_1)
                                + vt_1/Lf * deltat_1  * dt);

                } // End Evaluvate constraints at t.
        } // End Operator()
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
        bool ok = true;
        typedef CPPAD_TESTVECTOR(double) Dvector;

        // Note on why this flattaning rather than using objects for state etc.,
        // The state and coeff values are kept in flat list to store
        // predicted values so that previous values can be used
        // effectively without re-calculation; Bottom-up dynamic prograaming
        // technique. Since the prediction goes from latest car state at t0, the
        // values predicted at t1 is probably more accurate than tN. So I believe
        // top-down DP might not be as effecient in getting better results even
        // though it has better performace.

        // Initial values.
        double x_init = state[0];
        double y_init = state[1];
        double psi_init = state[2];
        double v_init = state[3];
        double cte_init = state[4]; // Cross Track error.
        double epsi_init = state[5]; // Error in psi.

        // Include states and actuators. All elements are flattened
        // state = (x,y, psi, v, cte, epsi)
        // actuators = {steering_delta, acceleration)
        // For example: If the state is a 4 element vector, the actuators is 2
        // element vector and there are 10 timesteps. The number of variables:
        // 4 * 10 + 2 * 9
        size_t n_vars = N * state.size() + (N-1) * 2;
        // Set the number of constraints
        size_t n_constraints = N * state.size();

        // Initial value of the independent variables.
        Dvector vars(n_vars);
        for (int i = 0; i < n_vars; i++) {
                vars[i] = 0;
        } // XXXMM: This means that in overload () in FG_Eval could
        // utilize this info and only compute when values are non-zero.

        // Update inital values, i.e., t = 0 values.
        vars[x_start] = x_init;
        vars[y_start] = y_init;
        vars[psi_start] = psi_init;
        vars[v_start] = v_init;
        vars[cte_start] = cte_init;
        vars[epsi_start] = epsi_init;

        Dvector vars_lowerbound(n_vars);
        Dvector vars_upperbound(n_vars);
        // Initialize non-actuators. Actuators start at delta_start.
        for (int i=0; i<delta_start; i++) {
                vars_lowerbound[i] = -1.0e19; // Give lowest value initially.
                vars_upperbound[i] = 1.0e19;
        }

        // Initialize delta to be between [-25, 25] degrees.
        for (int i = delta_start; i < a_start; i++) {
                vars_lowerbound[i] = -0.436332;
                vars_upperbound[i] = 0.436332;
        }

        // Initialize acceleration [-1 1]
        for(int i = a_start; i < n_vars; i++) {
                vars_lowerbound[i] = -1.0;
                vars_upperbound[i] = 1.0;
        }

        // Lower and upper limits for the constraints
        // Should be 0 besides initial state.
        Dvector constraints_lowerbound(n_constraints);
        Dvector constraints_upperbound(n_constraints);
        for (int i = 0; i < n_constraints; i++) {
                constraints_lowerbound[i] = 0;
                constraints_upperbound[i] = 0;
        }
       // Initial values
        constraints_lowerbound[x_start] = x_init;
        constraints_lowerbound[y_start] = y_init;
        constraints_lowerbound[psi_start] = psi_init;
        constraints_lowerbound[v_start] = v_init;
        constraints_lowerbound[cte_start] = cte_init;
        constraints_lowerbound[epsi_start] = epsi_init;

        // For upper bounds.
        constraints_upperbound[x_start] = x_init;
        constraints_upperbound[y_start] = y_init;
        constraints_upperbound[psi_start] = psi_init;
        constraints_upperbound[v_start] = v_init;
        constraints_upperbound[cte_start] = cte_init;
        constraints_upperbound[epsi_start] = epsi_init;

        // object that computes objective and constraints
        FG_eval fg_eval(coeffs);

        // options for IPOPT solver
        std::string options;
        // Uncomment this if you'd like more print information
        options += "Integer print_level  0\n";
        // NOTE: Setting sparse to true allows the solver to take advantage
        // of sparse routines, this makes the computation MUCH FASTER. If you
        // can uncomment 1 of these and see if it makes a difference or not but
        // if you uncomment both the computation time should go up in orders of
        // magnitude.
        options += "Sparse  true        forward\n";
        options += "Sparse  true        reverse\n";
        // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
        // Change this as you see fit.
        options += "Numeric max_cpu_time          0.5\n";

        // place to return solution
        CppAD::ipopt::solve_result<Dvector> solution;

        // solve the problem
        CppAD::ipopt::solve<Dvector, FG_eval>(
            options, vars, vars_lowerbound, vars_upperbound,
            constraints_lowerbound, constraints_upperbound, fg_eval, solution);

        // Check some of the solution values
        ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

        // Cost
        auto cost = solution.obj_value;
        std::cout << "Cost " << cost << std::endl;

        // Predicted values. {delta, a, x0, y0, x1, y1, ..xN, yN}
        // delta: steering angle
        // a: acceleration
        // x1..N: predicted X values aka: green line
        // y1..N: predicted Y values
        // Miss python.
        std::vector<double> predicted = {solution.x[delta_start],
            solution.x[a_start]};

        for (int i = 0; i < N; i++) {
                predicted.push_back(solution.x[x_start + i]);
                predicted.push_back(solution.x[y_start + i]);
        }

        return predicted;
}
