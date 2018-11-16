#include "ParticleDisplay.h"

ParticleDisplay::ParticleDisplay(){}

void ParticleDisplay::init(const std::vector<Particle*>& particles)
{
	vertices_.resize(particles.size());

	// Setup Vertex Array Object and Vertex Buffer Object
	glGenVertexArrays(1, &VAO_);
	glGenBuffers(1, &VBO_);

	updateParticles(particles);
}

void ParticleDisplay::updateParticles(const std::vector<Particle*>& particles, double xExtent, double yExtent)
{
	// Update vertices
	for (const auto particle : particles)
	{
		vertices_[particle->getDispIdx()].x = particle->getPos().x / xExtent;
		vertices_[particle->getDispIdx()].y = particle->getPos().y / yExtent;
	}

	// Bind the Vertex Array Object first, then bind and set vertex buffer(s) and attribute pointer(s).
	glBindVertexArray(VAO_);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_);
	glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof(glm::vec2), &vertices_.front(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);

	// Unbind buffers
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0); 
}


ParticleDisplay::~ParticleDisplay()
{
}

void ParticleDisplay::draw(GLenum drawMode)
{
	glBindVertexArray(VAO_);
	glDrawArrays(drawMode, 0, vertices_.size());
	glBindVertexArray(0);
}
