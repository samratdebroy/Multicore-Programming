#pragma once
// TODO: Should probably put these constants in a namespace
constexpr double SIM_SIZE = 1.0e7; // Vertical and Horizontal extent of the simulation screen in units of length
constexpr double MAX_GALAXY_RADIUS = SIM_SIZE * 0.3;
constexpr double MIN_GALAXY_RADIUS = MAX_GALAXY_RADIUS * 0.05;
constexpr double PARTICLE_MASS = 1.0e15;
constexpr double GRAVITATIONAL_CONSTANT = 6.67408e-11;
constexpr double THETA = 1.0;
constexpr double SOFTENER = 0.0001*(MAX_GALAXY_RADIUS)*(MAX_GALAXY_RADIUS);
constexpr double TIME_STEP = 0.2*60*60;

constexpr int NUM_GALAXIES = 2;
constexpr int NUM_PARTICLES = 64* 64;
constexpr int NUM_NODES = 4 * NUM_PARTICLES;

constexpr int BLOCKSIZE = 64;
constexpr int WARP_SIZE = 32;
constexpr int MAX_DEPTH = 64; // Max depth of the stack,  defines how many levels of recursion we can have
