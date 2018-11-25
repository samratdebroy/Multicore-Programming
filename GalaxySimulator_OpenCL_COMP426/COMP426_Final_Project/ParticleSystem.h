#pragma once
#include <vector>
#include <array>

#include "Particle.h"
#include "ParticleDisplay.h"
#include "BHQuadtreeNode.h"
#include "SimulationConstants.h"
#include "OpenCLManager.h"


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
	cl_float4 min_max_extents[1] = { { FLT_MAX, FLT_MAX , -FLT_MAX, -FLT_MAX } };
	cl_float4 getMinMaxSerial();
	OpenCLManager openCLManager;

public:
	ParticleSystem(HGLRC& openGLContext, HDC& hdc);
	void draw(int galaxyIndex, GLenum drawMode = GL_POINTS);
	void performComputations();
	void resetQuadtreeSerial();
	~ParticleSystem();
};