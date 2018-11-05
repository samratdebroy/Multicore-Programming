#pragma once
#include <array>
#include <memory>
#include "Particle.h"


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

	int nodeID_;
	float mass_;
	float2 centerOfMass_;
	float2 min_; // Lower left edge of node
	float2 max_; // Upper right edge of node
	float2 center_;
	BHQuadtreeNode* parent_;
	Particle* particle_;
	unsigned int numParticles_;

	bool isRoot() const;
	bool isLeaf() const;

	Quadrant getQuadrant(const Particle* const particle);

public:
	static int nodeID_counter;

	BHQuadtreeNode(int nodeID, const float2& min, const float2& max, BHQuadtreeNode* parent = nullptr);
	std::array<std::unique_ptr<BHQuadtreeNode>, 4> quadrants;

	void insertParticle(Particle* const particle);
	void computeMassDistribution();
	void copyToArray(float * mass, int * child, float2 * pos, int depth = 0);

	//Getters and Setters
	int getNodeID() const { return nodeID_; }
	float getMass() const { return mass_; }
	const float2& getMin() const { return min_; }
	const float2& getMax() const { return max_; }
	const float2& getCenter() const { return center_; }
	const float2& getCenterOfMass() const { return centerOfMass_; }
	const unsigned int getNumParticles() const { return numParticles_; }
	const Particle* getParticle() const { return particle_; }

};