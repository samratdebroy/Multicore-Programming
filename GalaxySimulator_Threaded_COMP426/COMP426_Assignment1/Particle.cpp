#include "Particle.h"



Particle::Particle(int idx, float mass) : idx_(idx), mass_(mass)
{
	pos_ = glm::vec2(0.0f, 0.0f);
	vel_ = glm::vec2(0.0f, 0.0f);
	acc_ = glm::vec2(0.0f, 0.0f);

}


Particle::~Particle()
{
}
