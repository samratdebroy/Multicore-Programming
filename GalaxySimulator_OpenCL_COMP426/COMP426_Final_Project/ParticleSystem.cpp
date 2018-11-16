#include "kernels.cl"
#include <random>
#include <limits>

#include "ParticleSystem.h"

// #define PROFILE true // Uncomment to profile
#ifdef PROFILE
#include <iostream>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;
#endif

ParticleSystem::ParticleSystem(unsigned int numParticles)
{
	// TODO: remove hack
	numParticles = NUM_PARTICLES;

	particles_.reserve(numParticles);

	// Random Number Generator
	std::mt19937 rng;
	rng.seed(std::random_device()());

	// Separate particles by galaxies, init with norm dist around different pts per galaxy
	const int numGalaxies = galaxies_.size();
	int idx = -1;
	for (int i = 0; i < numGalaxies; i++)
	{
		int galaxySize;
		if (i < numGalaxies - 1 && numGalaxies > 1)
		{
			// Assign a random portion of the particles to this galaxy
			std::uniform_real_distribution<double> randDouble(0.3 / (numGalaxies - 1), 0.7 / (numGalaxies - 1));
			galaxySize = numParticles * randDouble(rng);
			numParticles -= galaxySize;
		}
		else
		{
			galaxySize = numParticles;
		}

		// Get an initial center point for the galaxy around which particles will spawn
		std::uniform_real_distribution<float> randFloat(-0.6*SIM_SIZE, 0.6*SIM_SIZE);
		float2 galaxyCenter = {randFloat(rng), randFloat(rng)};

		// Use a uniform dist to get angle and distance from center
		std::uniform_real_distribution<double> randAngle(0, 2 * 3.141592653); // distribution returns angle between 0 and 2*PI rads
		std::uniform_real_distribution<double> randDistFromCenter(0.0, MAX_GALAXY_RADIUS);

		// Create particles
		galaxies_[i].particles.reserve(galaxySize);
		for (int j = 0; j < galaxySize; j++)
		{
			++idx;
			mass_[idx] = PARTICLE_MASS;
			particles_.push_back(Particle(idx, j, mass_, pos_, vel_, acc_));
			galaxies_[i].particles.push_back(&particles_.back());

			// Calculate point's position relative to galaxy center
			auto const& angle = randAngle(rng); // Angle from center at which particle will spawn
			auto const& dist = randDistFromCenter(rng); // distance from center at which particle will spawn
			float2 pos = {glm::cos(angle)*dist, glm::sin(angle)*dist};
			// Move it's position away from the center by the minimum galaxy radius
			float2 distance_from_center = { glm::normalize(glm::vec2(pos.x, pos.y)).x*MIN_GALAXY_RADIUS,
											glm::normalize(glm::vec2(pos.x, pos.y)).y*MIN_GALAXY_RADIUS};
			pos.x = pos.x + distance_from_center.x + galaxyCenter.x;
			pos.y = pos.y + distance_from_center.y + galaxyCenter.y;
			galaxies_[i].particles[j]->setPos(pos);

			auto const& speed = PARTICLE_MASS * GRAVITATIONAL_CONSTANT / (dist) *50;
			float2 vel = { glm::sin(angle)*speed, -glm::cos(angle)*speed };
			galaxies_[i].particles[j]->setVel(vel);
		}

		// Connect these particles to the Particle Displayer
		galaxies_[i].particleDisplay.init(galaxies_[i].particles);
	}

	// Initialize the openCL kernels and devices
	openCLManager.init();

	// Make sure computations are performed at least once after initialization
	performComputations();
}

void ParticleSystem::draw(int galaxyIndex, GLenum drawMode)
{
	const auto& xExtent = SIM_SIZE;
	const auto& yExtent = SIM_SIZE;
	galaxies_[galaxyIndex].particleDisplay.updateParticles(galaxies_[galaxyIndex].particles, xExtent, yExtent);
	galaxies_[galaxyIndex].particleDisplay.draw(drawMode);
}

void ParticleSystem::performComputations()
{
#ifdef PROFILE
	auto t1 = Clock::now();
#endif
	// 1. Find the Min/Max values of the frame
	std::pair<float2, float2> minmax;
	minmax.first = { FLT_MAX, FLT_MAX };
	minmax.second = { -FLT_MAX, -FLT_MAX };
	for (auto& particle : particles_)
	{
		auto& min = minmax.first;
		auto& max = minmax.second;

		const auto& pos = particle.getPos();
		if (pos.x < min.x)
			min.x = pos.x;
		if (pos.y < min.y)
			min.y = pos.y;
		if (pos.x > max.x)
			max.x = pos.x;
		if (pos.y > max.y)
			max.y = pos.y;
	}
	float4 min_max_extents = { minmax.first.x, minmax.first.y, minmax.second.x, minmax.second.y};

#ifdef PROFILE
	std::cout << "1. Min Max calculations: "
		<< std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t1).count()
		<< " microseconds" << std::endl;
	t1 = Clock::now();
#endif

	// 2. Create root for quad tree
	BHQuadtreeNode::nodeID_counter = 0;
	BHQuadtreeNode root(0, minmax.first, minmax.second,nullptr);

	// 3. Build tree by inserting all the particles
	for (auto& particle : particles_)
	{
		root.insertParticle(&particle);
	}

#ifdef PROFILE
	std::cout << "2. and 3. Create root and build tree: "
		<< std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t1).count()
		<< " microseconds" << std::endl;
	t1 = Clock::now();
#endif

	// 4. Compute mass distribution
	root.computeMassDistribution();

#ifdef PROFILE
	std::cout << "4. mass distribution: "
		<< std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t1).count()
		<< " microseconds" << std::endl;
	t1 = Clock::now();
#endif 

	// 5. Copy values computed for each node into an array so it can be used with CUDA
	//openCLManager.resetQuadtree(pos_, mass_, child_);
	resetQuadtreeSerial();
	root.copyToArray(mass_, child_, pos_);

#ifdef PROFILE
	std::cout << "5. copy quadtree values into arrays: "
		<< std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t1).count()
		<< " microseconds" << std::endl;
	t1 = Clock::now();
#endif

	//openCLManager.computeForcesAndIntegrate(pos_, vel_, acc_, mass_, child_, &min_max_extents);
	openCLManager.computeForcesAndIntegrate(pos_, vel_, acc_, mass_);

#ifdef PROFILE
	std::cout << "6. compute forces and integrate: "
		<< std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t1).count()
		<< " microseconds" << std::endl;
#endif
}

void ParticleSystem::resetQuadtreeSerial()
{
	for (int nodeID = 0; nodeID < NUM_NODES; ++nodeID)
	{
		// Reset all child node indices to -1 (Null)
		for (int i = 0; i < 4; ++i)
		{
			child_[nodeID * 4 + i] = -1;
		}

		// Only reset mass and position for internal nodes
		if (nodeID >= NUM_PARTICLES)
		{
			pos_[nodeID] = { 0.0f, 0.0f };
			mass_[nodeID] = 0.0f;
		}
	}
}

ParticleSystem::~ParticleSystem()
{
}