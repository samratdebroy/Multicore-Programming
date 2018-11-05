#include "cuda_computation.cuh"
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
			std::uniform_real_distribution<double> randDouble(0.3/(numGalaxies-1), 0.7/(numGalaxies-1));
			galaxySize = numParticles * randDouble(rng);
			numParticles -= galaxySize;
		}
		else
		{
			galaxySize = numParticles;
		}

		// Get an initial center point for the galaxy around which particles will spawn
		std::uniform_real_distribution<double> randDouble(-0.6*SIM_SIZE,0.6*SIM_SIZE);
		auto galaxyCenter = make_float2(randDouble(rng), randDouble(rng));
		
		// Use a uniform dist to get angle and distance from center
		std::uniform_real_distribution<double> randAngle(0, 2*3.141592653); // distribution returns angle between 0 and 2*PI rads
		std::uniform_real_distribution<double> randDistFromCenter(0.0, MAX_GALAXY_RADIUS);

		// Create particles
		galaxies_[i].particles.reserve(galaxySize);
		for (int j = 0; j < galaxySize; j++)
		{
			++idx;
			mass_[idx] = PARTICLE_MASS;
			particles_.push_back( Particle(idx, j, mass_, pos_, vel_, acc_) );
			galaxies_[i].particles.push_back(&particles_.back());

			// Calculate point's position relative to galaxy center
			auto const& angle = randAngle(rng); // Angle from center at which particle will spawn
			auto const& dist = randDistFromCenter(rng); // distance from center at which particle will spawn
			auto pos = make_float2(glm::cos(angle)*dist, glm::sin(angle)*dist);
			// Move it's position away from the center by the minimum galaxy radius
			pos = pos + make_float2(glm::normalize(glm::vec2(pos.x, pos.y)).x*MIN_GALAXY_RADIUS, glm::normalize(glm::vec2(pos.x, pos.y)).y*MIN_GALAXY_RADIUS);
			galaxies_[i].particles[j]->setPos(pos + galaxyCenter);

			auto const& speed = PARTICLE_MASS * GRAVITATIONAL_CONSTANT / (dist) *50;
			auto vel = make_float2(glm::sin(angle)*speed, -glm::cos(angle)*speed);
			galaxies_[i].particles[j]->setVel(vel);
		}

		// Connect these particles to the Particle Displayer
		galaxies_[i].particleDisplay.init(galaxies_[i].particles);
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	auto cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

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
	minmax.first = make_float2(FLT_MAX, FLT_MAX);
	minmax.second = make_float2(-FLT_MAX, -FLT_MAX);
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
	float4 min_max_extents = make_float4(minmax.first.x, minmax.first.y, minmax.second.x, minmax.second.y);

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

	// 4b. Copy values computed for each node into an array so it can be used with CUDA
	cudaError_t cudaStatus = reset_quadtree_with_cuda(pos_, mass_, child_);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "compute_forces_and_integrate_with_cuda failed!");
		return;
	}
	root.copyToArray(mass_, child_, pos_);

	// 5. Compute acceleration of each particle due to external forces
	//for (size_t i = 0; i < particles_.size(); ++i)
	//{
	//	particles_[i].setAcc(root.computeForceFromNode(&particles_[i]));
	//}


#ifdef PROFILE
	std::cout << "5. compute acceleration: "
		<< std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t1).count()
		<< " microseconds" << std::endl;
	t1 = Clock::now();
#endif

	// 6. Update the position and velocity of each particle
	//for (auto& particle : particles_)
	//	integrate(TIME_STEP, &particle);

	cudaStatus = compute_forces_and_integrate_with_cuda(TIME_STEP, pos_, vel_, acc_, mass_, child_, &min_max_extents);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "compute_forces_and_integrate_with_cuda failed!");
		return;
	}

#ifdef PROFILE
	std::cout << "6. integrate: "
		<< std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t1).count()
		<< " microseconds" << std::endl;
#endif
}

void ParticleSystem::integrate(double dt, Particle* particle)
{
	// Integrate velocity and position of all particles
	const auto& acc = particle->getAcc();
	particle->setVel(particle->getVel() +acc*dt);
	const auto& vel = particle->getVel();
	particle->setPos(particle->getPos() + vel*dt);
}

ParticleSystem::~ParticleSystem()
{
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	auto cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return;
	}
}