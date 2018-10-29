#pragma once
#include "glm/glm.hpp"
#include <vector_types.h>
#include <vector>

class Particle
{
private:
	int idx_;
	int disp_idx_;
	double* mass_;
	// position, velocity and acceleration
	float2* pos_;
	float2* vel_;
	float2* acc_;

public:
	Particle(int id, int disp_id, double* mass, float2* pos, float2* vel, float2* acc);
	~Particle();

	// Setters and Getters
	int getIdx() const { return idx_; }
	int getDispIdx() const { return disp_idx_; }
	double getMass() const { return mass_[idx_]; }
	void setMass(double newMass) { mass_[idx_] = newMass; }
	const float2& getPos() const { return pos_[idx_]; }
	void setPos(const float2& newPos) { pos_[idx_] = newPos; }
	const float2& getVel() const { return vel_[idx_]; }
	void setVel(const float2& newVel) { vel_[idx_] = newVel; }
	const float2& getAcc() const { return acc_[idx_]; }
	void setAcc(const float2& newAcc) { acc_[idx_] = newAcc; }
};

