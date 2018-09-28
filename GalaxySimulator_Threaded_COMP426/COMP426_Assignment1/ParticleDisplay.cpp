#include "ParticleDisplay.h"



ParticleDisplay::ParticleDisplay(const std::vector<Particle>& particles)
{
	vertices.resize(particles.size());

	// Setup Vertex Array Object and Vertex Buffer Object
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	updateParticles(particles);
}

void ParticleDisplay::updateParticles(const std::vector<Particle>& particles)
{
	// Update vertices
	for (Particle particle : particles)
	{
		vertices[particle.getIdx()] = particle.getPos();
	}

	// Bind the Vertex Array Object first, then bind and set vertex buffer(s) and attribute pointer(s).
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices.front(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
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
	glBindVertexArray(VAO);
	glDrawArrays(drawMode, 0, vertices.size());
	glBindVertexArray(0);
}
