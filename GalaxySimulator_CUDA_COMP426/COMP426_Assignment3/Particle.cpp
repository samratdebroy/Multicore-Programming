#include "Particle.h"

Particle::Particle(int id, int disp_id, float* mass, float2* pos, float2* vel, float2* acc)
	: idx_(id), dispIdx_(disp_id), mass_(mass), pos_(pos), vel_(vel), acc_(acc)
{
}


Particle::~Particle()
{
}
