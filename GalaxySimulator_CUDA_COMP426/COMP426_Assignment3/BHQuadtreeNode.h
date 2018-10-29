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

	double mass_;
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
	float2 computeGravityAcc(const Particle* const p1, const Particle* const p2);

public:

	BHQuadtreeNode(const float2& min, const float2& max, BHQuadtreeNode* parent = nullptr);
	std::array<std::unique_ptr<BHQuadtreeNode>, 4> quadrants;

	void insertParticle(Particle* const particle);
	void computeMassDistribution();
	float2 computeForceFromNode(const Particle* const target_p);


	//Getters and Setters
	double getMass() const { return mass_; }
	const float2& getMin() const { return min_; }
	const float2& getMax() const { return max_; }
	const float2& getCenter() const { return center_; }
	const float2& getCenterOfMass() const { return centerOfMass_; }

};