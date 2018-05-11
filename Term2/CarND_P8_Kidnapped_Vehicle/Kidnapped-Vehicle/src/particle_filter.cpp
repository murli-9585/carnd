/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <stdlib.h>
#include "particle_filter.h"

using namespace std;
using std::normal_distribution;
using std::default_random_engine;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

        double init_weight = 1.0;
        // Set the number of particles.
        // XXXMM: The error rate when # is 100 is more than #200
        // but the difference between 200 and 300 is not much.
        // Still getting error on x and y ~ 0.1
        num_particles = 200;

        // Random noise
        default_random_engine gen;

        // Initialize all particles to first position (based on estimates of
        //   x, y, theta and their uncertainties from GPS).
        // std is expected to be of form [x, y, theta]
        normal_distribution<double> dist_x(0, std[0]);
        normal_distribution<double> dist_y(0, std[1]);
        normal_distribution<double> dist_theta(0, std[2]);

        // Update all particles to gaussian distrubtion around first position.
        // by adding some noise; Now all particles are spread across inital
        // positions.
        for (int i=0; i<num_particles; ++i) {
                Particle p;
                p.id = i;
                p.x = x + dist_x(gen);
                p.y = y + dist_y(gen);
                p.theta = theta + dist_theta(gen);
                p.weight = init_weight;

                particles.push_back(p);
                weights.push_back(init_weight);
        }

        is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
        double velocity, double yaw_rate) {
        //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
        //  http://www.cplusplus.com/reference/random/default_random_engine/

        default_random_engine gen;

        // Normalize x, y and angle positions.
        normal_distribution<double> x_init(0, std_pos[0]);
        normal_distribution<double> y_init(0, std_pos[1]);
        normal_distribution<double> theta_init(0, std_pos[2]);

        // Predict each particle's new position based on given values.
        for (int i=0; i<particles.size(); i++) {
                Particle p = particles[i];
                // Avoid dividing by 0
                if (fabs(yaw_rate) < 0.00001) {
                        p.x = p.x + velocity * delta_t * cos(p.theta);
                        p.y = p.y + velocity * delta_t * sin(p.theta);
                        // No theta update needed.
                }
                else {
                       p.x = p.x + velocity * (sin(p.theta + yaw_rate*delta_t)
                                               - sin(p.theta)) / yaw_rate;
                       p.y = p.y + velocity * (cos(p.theta) - cos(p.theta +
                                               yaw_rate*delta_t)) / yaw_rate;
                       p.theta = p.theta + yaw_rate*delta_t;
                }
                // Normalize.
                p.x = p.x + x_init(gen);
                p.y = p.y + y_init(gen);
                p.theta = p.theta + theta_init(gen);

                particles[i] = p;
        }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
            std::vector<LandmarkObs>& observations) {
        // Find the predicted measurement that is closest to each observed
        // measurement and assign the
        // observed measurement to this particular landmark.

       double current_min = INFINITY; // Double max in math.h.
       double distance_gap;
       int min_index;

       for (int i_obs=0; i_obs < observations.size(); ++i_obs) {
            LandmarkObs obs = observations[i_obs];
            // Re-initialize.
            current_min = INFINITY;
            min_index = -1;
            for (int i_pred=0; i_pred < predicted.size(); ++i_pred) {
                    LandmarkObs pred = predicted[i_pred];
                    distance_gap = dist(obs.x, obs.y, pred.x, pred.y);
                    if (distance_gap < current_min) {
                            current_min = distance_gap;
                            min_index = i_pred;
                    }
            }
            //  Track the index of minimum distance prediction point.
            observations[i_obs].id = min_index;
       }
}

LandmarkObs ParticleFilter::convertToMapCoordinates(const LandmarkObs &obs,
                            const Particle &p) {
        // Converts observations noted on local/simulator to map
        // co-ordinate system.
        // Ref: The following is a good resource for the theory:
        // www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
        // Equation: 3.33 of http://planning.cs.uiuc.edu/node99.html
        LandmarkObs lm_updated;

        lm_updated.x = p.x + obs.x*cos(p.theta) - obs.y*sin(p.theta);
        lm_updated.y = p.y + obs.x*sin(p.theta) + obs.y*cos(p.theta);
        lm_updated.id = obs.id;
        return lm_updated;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
        const std::vector<LandmarkObs> &observations,
        const Map &map_landmarks) {
        // Update the weights of each particle using a mult-variate
        // Gaussian distribution.
        // https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        // NOTE: The observations are given in the VEHICLE'S coordinate system.
        // Particles are located
        // according to the MAP'S coordinate system.

        double std_dev_x_sqr = pow(std_landmark[0], 2);
        double std_dev_y_sqr = pow(std_landmark[1], 2);
        double gauss_norm = 1.0 / (2.0*M_PI*std_landmark[0]*std_landmark[1]);
        double weight = 1.0; // Its probabilistic weight.
        double dx_sqr, dy_sqr; // Distance between observed and predicted.

        // Itertate over particles and update particles weight
        // and weights vector.
        for (int i_p=0; i_p<particles.size(); ++i_p) {
                Particle p = particles[i_p];

                //Step 1: Get transformed observations.
                // Convert to global map system.
                std::vector<LandmarkObs> observations_t;
                for (LandmarkObs lm_obs : observations) {
                        LandmarkObs obs_t = convertToMapCoordinates(lm_obs, p);
                        observations_t.push_back(obs_t);
                }

                // Step 2: Find closest landmarks to particle.
                std::vector<LandmarkObs> landmarks_pred;
                // Distance between landmark on map to particle.
                double landmark_to_particle;

                // Only update predicted landmarks which fall in the range.
                for (Map::single_landmark_s m_lm : map_landmarks.landmark_list) {
                        landmark_to_particle = dist(p.x, p.y,
                                                m_lm.x_f, m_lm.y_f);
                        // Include landmarks with-in range;
                        if (landmark_to_particle < sensor_range) {
                                LandmarkObs landmark_pred;

                                landmark_pred.x = m_lm.x_f;
                                landmark_pred.y = m_lm.y_f;
                                landmark_pred.id = m_lm.id_i;

                                landmarks_pred.push_back(landmark_pred);
                        }
                }

                // Step 3: Associate predicted LM id to observations.
                // Store the predicted index which satisfy minimum distance
                // criteria in transformed index.
                // Every observsation will have a nearest/closest
                // Predicted landmark.
               dataAssociation(landmarks_pred, observations_t);

                // Step 4: Calculate multivariate Gaussian for observed
               // landmark and predicited landmarks.
                // Calculate gaussian for each transformed observed landmark
                // and associated predicted landmark.
                weight = 1.0;
                for (int i=0; i<observations_t.size(); ++i) {
                        LandmarkObs observed = observations_t[i];
                        // predicted lm closest to observed lm.
                        // Note: observed.id has the index which is close.
                        LandmarkObs pred = landmarks_pred[observed.id];

                        dx_sqr = pow((observed.x - pred.x), 2);
                        dy_sqr = pow((observed.y - pred.y), 2);
                        weight *= gauss_norm * exp(-(dx_sqr/(2*std_dev_x_sqr) +
                                      dy_sqr/(2*std_dev_y_sqr)));
                }
                // Step 5: Update weights with probability.
                particles[i_p].weight = weight;
                weights[i_p] = weight;

        } // Particle iteration.
}

void ParticleFilter::resample_discreet() {
        // Resample particles with replacement with probability proportional
        // to their weight.
        // std::discrete_distribution helpful here.
        // http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
        default_random_engine gen;

        discrete_distribution<int> dist(weights.begin(), weights.end());
        vector<Particle> new_particles(particles.size());

        for(int i=0; i<particles.size(); ++i) {
                new_particles[i] = particles[dist(gen)];
                weights[i] = particles[dist(gen)].weight;
        }
        particles = new_particles;
}

void ParticleFilter::resample() {
    resample_discreet();
}
// Needs to fix.
void ParticleFilter::resample_wheel() {
        // Resample the particles using resampling wheel method.
        // Ref: Sabastien example in particle filter.
        // The particles are picked according to importance of weights.
        // Conversly Pick particles according to importance but give
        // randomness to the particle distribution.

        default_random_engine gen;
        vector<Particle> new_particles(particles.size());
        uniform_real_distribution<double> dist(0.0, 1.0);

        int current_index = rand() % num_particles + 1;
        cout << "CURRENT INDEX: " << current_index << endl;

        double max_weight = 0.0; //= *max_element(begin(weights), end(weights));
        for (auto it : weights) {
            if (it >= max_weight)
                    max_weight = it;
        }
        // If the max_weight is < 0.001 our wheel has weights
        // that are no so useful. Lets use discreet sampling in such scenario.
        if (max_weight < 0.001) {
                resample_discreet();
                return;
        }
        cout << " MAX WEIGHT: " << max_weight << endl;

        double beta = 0;
        for (int i=0; i<particles.size(); ++i) {
                double random = ((double)rand()) / ((double)RAND_MAX);
                beta += random*2.0*max_weight;
                while ( particles[current_index].weight < beta) {
                        beta -= particles[current_index].weight;
                        current_index = (current_index + 1) % num_particles;
                }
                new_particles[current_index] = particles[current_index];
                weights[current_index] = particles[current_index].weight;

        }
        particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle,
        std::vector<int> associations, std::vector<double> sense_x,
        std::vector<double> sense_y)
{
        //particle: the particle to assign each listed association,
        // and association's (x,y) world coordinates mapping to
        // associations: The landmark id that goes along with each
        // listed association
        // sense_x: the associations x mapping already converted to
        // world coordinates
        // sense_y: the associations y mapping already converted to
        // world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
