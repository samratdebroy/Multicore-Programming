#include "ParticleSystem.h"

#include "tbb\tbb.h"
#include "tbb\parallel_reduce.h"
#include "tbb\blocked_range.h"
#include "tbb\parallel_for.h"

#include "SimulationConstants.h"

#include <random>
#include <limits>

// #define PROFILE true // Uncomment to profile
#ifdef PROFILE
#include <iostream>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;
#endif

ParticleSystem::ParticleSystem(unsigned int numParticles)
{
	particles_.reserve(numParticles);

	// Random Number Generator
	std::mt19937 rng;
	rng.seed(std::random_device()());

	// Separate particles by galaxies, init with norm dist around different pts per galaxy
	const int numGalaxies = galaxies_.size();
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
		auto galaxyCenter = glm::vec2(randDouble(rng), randDouble(rng));
		
		// Use a uniform dist to get angle and distance from center
		std::uniform_real_distribution<double> randAngle(0, 2*3.141592653); // distribution returns angle between 0 and 2*PI rads
		std::uniform_real_distribution<double> randDistFromCenter(0.0, MAX_GALAXY_RADIUS);

		// Create particles
		galaxies_[i].particles.reserve(galaxySize);
		for (unsigned int j = 0; j < galaxySize; j++)
		{
			particles_.push_back(Particle(j, PARTICLE_MASS));
			galaxies_[i].particles.push_back(&particles_.back());

			// Calculate point's position relative to galaxy center
			auto const& angle = randAngle(rng); // Angle from center at which particle will spawn
			auto const& dist = randDistFromCenter(rng); // distance from center at which particle will spawn
			auto pos = glm::vec2(glm::cos(angle)*dist, glm::sin(angle)*dist);
			// Move it's position away from the center by the minimum galaxy radius
			pos = pos + glm::vec2(glm::normalize(pos).x*MIN_GALAXY_RADIUS, glm::normalize(pos).y*MIN_GALAXY_RADIUS);
			galaxies_[i].particles[j]->setPos(pos + galaxyCenter);

			auto const& speed = PARTICLE_MASS * GRAVITATIONAL_CONSTANT / (dist);
			auto vel = glm::vec2(glm::sin(angle)*speed, -glm::cos(angle)*speed);
			galaxies_[i].particles[j]->setVel(vel);
		}

		// Connect these particles to the Particle Displayer
		galaxies_[i].particleDisplay.init(galaxies_[i].particles);
	}
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
	auto getMinMax = [&](const tbb::blocked_range<size_t>& range, std::pair<glm::vec2, glm::vec2> minmax) -> auto
	{
		auto& min = minmax.first;
		auto& max = minmax.second;
		for (size_t i = range.begin(); i != range.end(); ++i)
		{
			const auto& pos = particles_[i].getPos();

			if (pos.x < min.x)
				min.x = pos.x;
			if (pos.y < min.y)
				min.y = pos.y;
			if (pos.x > max.x)
				max.x = pos.x;
			if (pos.y > max.y)
				max.y = pos.y;
		}
		return minmax;
	};

	std::pair<glm::vec2, glm::vec2> default_minmax;
	default_minmax.first = glm::vec2(FLT_MAX, FLT_MAX);
	default_minmax.second = glm::vec2(-FLT_MAX, -FLT_MAX);

	auto const minmax = tbb::parallel_reduce( tbb::blocked_range<size_t>(0, particles_.size()),
		default_minmax,
		getMinMax,
		[](std::pair<glm::vec2, glm::vec2> lhs, std::pair<glm::vec2, glm::vec2> rhs)->auto{
				// Get mins
				if (lhs.first.x < rhs.first.x)
					rhs.first.x = lhs.first.x;
				if (lhs.first.y < rhs.first.y)
					rhs.first.y = lhs.first.y;
				// Get maxes
				if (lhs.second.x > rhs.second.x)
					rhs.second.x = lhs.second.x;
				if (lhs.second.y > rhs.second.y)
					rhs.second.y = lhs.second.y;
			return rhs;
		}
	);

#ifdef PROFILE
	std::cout << "1. Min Max calculations: "
		<< std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t1).count()
		<< " microseconds" << std::endl;
	t1 = Clock::now();
#endif

	// 2. Create root for quad tree
	BHQuadtreeNode root(minmax.first, minmax.second,nullptr);

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

	// 5. Compute acceleration of each particle due to external forces
	tbb::parallel_for(tbb::blocked_range<size_t>(0, particles_.size()),
		[&](const tbb::blocked_range<size_t>& range) {
			for (size_t i = range.begin(); i != range.end(); ++i)
			{
				particles_[i].setAcc(root.computeForceFromNode(&particles_[i]));
			}
		}, // The body
		tbb::auto_partitioner() // The default partitioner
	);

#ifdef PROFILE
	std::cout << "5. compute acceleration: "
		<< std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t1).count()
		<< " microseconds" << std::endl;
	t1 = Clock::now();
#endif

	// 6. Update the position and velocity of each particle
	tbb::parallel_for(tbb::blocked_range<size_t>(0, particles_.size()),
		[&](const tbb::blocked_range<size_t>& range) {
			for (size_t i = range.begin(); i != range.end(); ++i)
			{
				integrate(TIME_STEP, &particles_[i]);
			}
		}, // The body
		tbb::auto_partitioner() // The default partitioner
		);

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
	particle->setVel(particle->getVel() + glm::vec2(acc.x*dt, acc.y*dt));
	const auto& vel = particle->getVel();
	particle->setPos(particle->getPos() + glm::vec2(vel.x*dt, vel.y*dt));
}

ParticleSystem::~ParticleSystem()
{
}