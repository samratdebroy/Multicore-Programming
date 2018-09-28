#pragma once
#include "glm.hpp"

class Particle
{
private:
	int idx;
	float mass;
	glm::vec3 pos;

public:
	Particle(int id = 1, float mass = 1.0f);
	~Particle();

	int getIdx() { return idx; }
	void setIdx(int newIdx) { idx = newIdx; }
	float getMass() { return mass; }
	void setMass(float newMass) { mass = newMass; }
	glm::vec3 getPos() { return pos; }
	void setPos(glm::vec3 newPos) { pos = newPos; }
};

