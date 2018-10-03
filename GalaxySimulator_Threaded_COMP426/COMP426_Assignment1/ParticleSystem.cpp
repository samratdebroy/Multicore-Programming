#include "ParticleSystem.h"

#include "SimulationConstants.h"

#include <random>
#include <limits>

namespace
{
	// Only spawn std::thread::hardware_concurrency() - 2 threads
	// bc one thread used for main and one for computation control
	const int NUM_THREADS = std::thread::hardware_concurrency() - 2;
}

ParticleSystem::ParticleSystem(unsigned int numParticles)
{
	particles_.reserve(numParticles);
	threadPool_.init(NUM_THREADS);

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
	// Find the Min/Max values of the frame
	glm::vec2 min(FLT_MAX , FLT_MAX);
	glm::vec2 max(-FLT_MAX , -FLT_MAX);
	for (const auto& particle : particles_)
	{
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

	// Create root for quad tree
	BHQuadtreeNode root(min,max,nullptr);

	// Build tree by inserting all the particles
	for (auto& particle : particles_)
	{
		root.insertParticle(&particle);
	}

	// Compute mass distribution
	root.computeMassDistribution();

	// Compute acceleration of each particle due to external forces
	std::vector<std::future<void>> results;
	results.reserve(particles_.size());
	for (auto& particle : particles_)
	{
		results.emplace_back(threadPool_.push([&particle, &root](){
				particle.setAcc(root.computeForceFromNode(&particle)); 
		}));
	}
	for (auto& result : results)
	{
		result.wait();
	}

	// Update the position and velocity of each particle ; split work evenly bw threads
	results.clear();
	results.reserve(NUM_THREADS);
	for (int i = 0; i < NUM_THREADS; i++)
	{
		results.emplace_back(threadPool_.push([this, i]() {
			int batchSize = particles_.size() / NUM_THREADS;
			int end = (i < NUM_THREADS - 1) ? batchSize*(i+1) : particles_.size();
			for (int j = batchSize * i; j < end; j++)
			{
				integrate(TIME_STEP, &particles_[j]);
			}
		}));
	}
	for (auto& result : results)
	{
		result.wait();
	}
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