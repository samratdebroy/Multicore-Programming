#pragma once
#include "glm/glm.hpp"
#include "glad/glad.h"
#include <string>
#include "Particle.h"

class ParticleDisplay
{

private:
	/* Render Data */
	unsigned int VAO_, VBO_;
	std::vector<glm::vec2> vertices_;
	int offset_ = 0;

public:
	ParticleDisplay();
	void init(const std::vector<Particle*>& particles, int offset);
	void updateParticles(const std::vector<Particle*>& particles, double xExtent = 1.0, double yExtent = 1.0);
	~ParticleDisplay();
	void draw(GLenum drawMode = GL_POINTS);
	int getOffset() const { return offset_; };
	std::pair<int, int> getOffsetAndSize() const;
	unsigned int getVBO() const { return VBO_; }
};

