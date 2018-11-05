#pragma once
#include <vector>
#include <array>

#include "Particle.h"
#include "ParticleDisplay.h"
#include "BHQuadtreeNode.h"
#include "SimulationConstants.h"

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
	float mass_[NUM_NODES];
	int child_[NUM_NODES * 4];
	float2 pos_[NUM_NODES];
	float2 vel_[NUM_PARTICLES];
	float2 acc_[NUM_PARTICLES];

public:
	ParticleSystem(unsigned int numParticles);
	void draw(int galaxyIndex, GLenum drawMode = GL_POINTS);
	void performComputations();
	~ParticleSystem();
};