#pragma once
#include <vector>
#include <array>
#include <memory>
#include "Particle.h"
#include "ParticleDisplay.h"

class ParticleSystem
{
private:

	// particles and display for two galaxies
	struct Galaxy
	{
		std::vector<Particle> particles;
		ParticleDisplay particleDisplay;
	};
	static constexpr int NUM_GALAXIES = 2;
	std::array<Galaxy, NUM_GALAXIES > galaxies_;

public:
	ParticleSystem(unsigned int numParticles);
	void draw(int galaxyIndex, GLenum drawMode = GL_POINTS);
	void performComputations();
	void integrate(double dt);
	~ParticleSystem();
};

// Barnes-Hut Quadtree implementation
class BHQuadtreeNode
{
private:

	enum class Quadrant : int
	{
		NE = 0,
		NW,
		SW,
		SE,
		NONE
	};

	double mass_;
	glm::vec2 centerOfMass_;
	glm::vec2 min_; // Lower left edge of node
	glm::vec2 max_; // Upper right edge of node
	glm::vec2 center_;
	BHQuadtreeNode* parent_;
	Particle* particle_;
	unsigned int numParticles_;

	bool isRoot() const;
	bool isLeaf() const;

	Quadrant getQuadrant(const Particle* const particle);
	glm::vec2 computeGravityAcc(const Particle* const p1, const Particle* const p2);

public:

	BHQuadtreeNode(const glm::vec2& min, const glm::vec2& max, BHQuadtreeNode* parent = nullptr);
	std::array<std::unique_ptr<BHQuadtreeNode>, 4> quadrants;

	void insertParticle(Particle* const particle);
	void computeMassDistribution();
	glm::vec2 computeForceFromNode(const Particle* const target_p);


	//Getters and Setters
	double getMass() const { return mass_; }
	//void setMass(double newMass) { mass_ = newMass; }
	const glm::vec2& getMin() const { return min_; }
	//void setMin(const glm::vec2& min) { min_ = min; }
	const glm::vec2& getMax() const { return max_; }
	//void setMax(const glm::vec2& max) { max_ = max; }
	const glm::vec2& getCenter() const { return center_; }
	//void setCenter(const glm::vec2& center) { center_ = center; }
	const glm::vec2& getCenterOfMass() const { return centerOfMass_; }
	//void setCenterOfMass(const glm::vec2& cm) { centerOfMass_ = cm; }

};