#pragma once
#include "glm.hpp"

class Particle
{
private:
	int idx_;
	float mass_;
	// position, velocity and acceleration
	glm::vec2 pos_;
	glm::vec2 vel_;
	glm::vec2 acc_;

public:
	Particle(int id = 1, float mass = 1.0f);
	~Particle();

	// Setters and Getters
	int getIdx() { return idx_; }
	void setIdx(int newIdx) { idx_ = newIdx; }
	float getMass() { return mass_; }
	void setMass(float newMass) { mass_ = newMass; }
	glm::vec2 getPos() { return pos_; }
	void setPos(glm::vec2 newPos) { pos_ = newPos; }
	glm::vec2 getVel() { return vel_; }
	void setVel(glm::vec2 newVel) { vel_ = newVel; }
	glm::vec2 getAcc() { return acc_; }
	void setAcc(glm::vec2 newAcc) { acc_ = newAcc; }
};

