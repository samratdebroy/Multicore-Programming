#pragma once
#include "glm.hpp"
#include <vector>
#include <string>
#include <glew.h>
#include "Particle.h"

class ParticleDisplay
{

private:
	/* Render Data */
	unsigned int VAO_, VBO_;
	std::vector<glm::vec2> vertices_;

public:
	ParticleDisplay(const std::vector<Particle>& particles);
	void updateParticles(const std::vector<Particle>& particles);
	~ParticleDisplay();
	void draw(GLenum drawMode = GL_POINTS);

};

