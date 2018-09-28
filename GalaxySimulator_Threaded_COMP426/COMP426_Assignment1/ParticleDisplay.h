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
	unsigned int VAO, VBO;
	GLuint vertices_VBO, normals_VBO;
	std::vector<glm::vec3> vertices;

public:
	ParticleDisplay(const std::vector<Particle>& particles);
	void updateParticles(const std::vector<Particle>& particles);
	~ParticleDisplay();
	void draw(GLenum drawMode = GL_POINTS);

};

