#pragma once

#include "tbb/task_scheduler_init.h"

#include <vector>
#include <array>

#include "Particle.h"
#include "ParticleDisplay.h"
#include "BHQuadtreeNode.h"

namespace {
	constexpr int NUM_GALAXIES = 2;
}

class ParticleSystem
{
private:

	// particles and display for two galaxies
	struct Galaxy
	{
		std::vector<Particle*> particles;
		ParticleDisplay particleDisplay;
	};
	std::array<Galaxy, NUM_GALAXIES > galaxies_;
	std::vector<Particle> particles_;
	tbb::task_scheduler_init task_scheduler;

public:
	ParticleSystem(unsigned int numParticles);
	void draw(int galaxyIndex, GLenum drawMode = GL_POINTS);
	void performComputations();
	void integrate(double dt, Particle* particle);
	~ParticleSystem();
};