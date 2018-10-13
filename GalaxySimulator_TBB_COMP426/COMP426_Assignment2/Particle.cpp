#include "Particle.h"

Particle::Particle(int idx, double mass) : idx_(idx), mass_(mass)
{
	pos_ = glm::vec2(0.0, 0.0);
	vel_ = glm::vec2(0.0, 0.0);
	acc_ = glm::vec2(0.0, 0.0);
}


Particle::~Particle()
{
}
