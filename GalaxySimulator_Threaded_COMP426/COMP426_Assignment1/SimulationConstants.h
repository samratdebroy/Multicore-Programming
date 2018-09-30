#pragma once
constexpr double SIM_SIZE = 1.0e7; // Vertical and Horizontal extent of the simulation screen in units of length
constexpr double MAX_GALAXY_RADIUS = SIM_SIZE * 0.3;
constexpr double MIN_GALAXY_RADIUS = MAX_GALAXY_RADIUS * 0.1;
constexpr double PARTICLE_MASS = 1.0e15;
constexpr double GRAVITATIONAL_CONSTANT = 6.67408e-11;
constexpr double THETA = 1.0;
constexpr double SOFTENER = (SIM_SIZE*3.0e-5)*(SIM_SIZE * 3.0e-5);
constexpr int NUM_THREADS = 8;
constexpr double TIME_STEP = 0.2*60*60;