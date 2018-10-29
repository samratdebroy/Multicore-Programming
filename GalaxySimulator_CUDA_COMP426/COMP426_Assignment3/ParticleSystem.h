#pragma once
#include <vector>
#include <array>

#include "Particle.h"
#include "ParticleDisplay.h"
#include "BHQuadtreeNode.h"

class ParticleSystem
{
private:

	static constexpr int NUM_GALAXIES = 2;
	static constexpr int NUM_PARTICLES = 5000;

	// particles and display for two galaxies
	struct Galaxy
	{
		std::vector<Particle*> particles;
		ParticleDisplay particleDisplay;
	};
	std::array<Galaxy, NUM_GALAXIES > galaxies_;
	std::vector<Particle> particles_;
	double mass_[NUM_PARTICLES];
	float2 pos_[NUM_PARTICLES];
	float2 vel_[NUM_PARTICLES];
	float2 acc_[NUM_PARTICLES];

public:
	ParticleSystem(unsigned int numParticles);
	void draw(int galaxyIndex, GLenum drawMode = GL_POINTS);
	void performComputations();
	void integrate(double dt, Particle* particle);
	~ParticleSystem();
};