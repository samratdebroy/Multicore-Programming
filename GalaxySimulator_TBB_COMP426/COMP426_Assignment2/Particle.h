#pragma once
#include "glm.hpp"

class Particle
{
private:
	int idx_;
	double mass_;
	// position, velocity and acceleration
	glm::vec2 pos_;
	glm::vec2 vel_;
	glm::vec2 acc_;

public:
	Particle(int id = 1, double mass = 1.0f);
	~Particle();

	// Setters and Getters
	int getIdx() const { return idx_; }
	void setIdx(int newIdx) { idx_ = newIdx; }
	double getMass() const { return mass_; }
	void setMass(double newMass) { mass_ = newMass; }
	const glm::vec2& getPos() const { return pos_; }
	void setPos(const glm::vec2& newPos) { pos_ = newPos; }
	const glm::vec2& getVel() const { return vel_; }
	void setVel(const glm::vec2& newVel) { vel_ = newVel; }
	const glm::vec2& getAcc() const { return acc_; }
	void setAcc(const glm::vec2& newAcc) { acc_ = newAcc; }
};

