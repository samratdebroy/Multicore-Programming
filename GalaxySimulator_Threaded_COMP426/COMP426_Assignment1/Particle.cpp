#include "Particle.h"



Particle::Particle(int idx, float mass) : idx(idx), mass(mass)
{
	pos = glm::vec3(0.0f, 0.0f, 0.0f);
}


Particle::~Particle()
{
}
