#include <cuda_helper\helper_math.h>

#include "BHQuadtreeNode.h"
#include "SimulationConstants.h"

int BHQuadtreeNode::nodeID_counter;

BHQuadtreeNode::BHQuadtreeNode(int nodeID, const float2& min, const float2 & max, BHQuadtreeNode * parent)
	: 
	nodeID_(nodeID),
	centerOfMass_(make_float2(0.0f, 0.0f)),
	min_(min),
	max_(max),
	center_(make_float2(min.x + (max.x - min.x) * 0.5f, min.y + (max.y - min.y) * 0.5f)),
	parent_(parent),
	particle_(nullptr),
	numParticles_(0)
{
	for (int i = 0; i < 4; i++)
	{
		quadrants[i].reset(nullptr);
	}
}

void BHQuadtreeNode::insertParticle(Particle * const particle)
{
	const auto& pos = particle->getPos();
	if ((pos.x < min_.x || pos.x > max_.x) || (pos.y < min_.y || pos.y > max_.y))
	{
		assert(false);
		// TODO: handle error
	}

	if (numParticles_ > 1)
	{
		// If there are more than one particles, insert it in the right quadrant
		const auto quadIndex = static_cast<std::underlying_type_t<Quadrant>>(getQuadrant(particle));
		quadrants[quadIndex]->insertParticle(particle);
	}
	else if (numParticles_ == 1)
	{
		// It is impossible for two particles to be in the same place
		if ((particle->getPos().x == particle_->getPos().x) && (particle->getPos().y == particle_->getPos().y))
		{
			assert(false);
			//TODO: Handle error
		}
		else
		{
			// Add quadrant nodes and relocate the particle
			const auto memberIdx = static_cast<std::underlying_type_t<Quadrant>>(getQuadrant(particle_));
			quadrants[memberIdx]->insertParticle(particle_);
			particle_ = nullptr;

			const auto paramIdx = static_cast<std::underlying_type_t<Quadrant>>(getQuadrant(particle));
			quadrants[paramIdx]->insertParticle(particle);
		}
	}
	else if (numParticles_ == 0)
	{
		particle_ = particle;
	}

	numParticles_++;
}

void BHQuadtreeNode::computeMassDistribution()
{
	if (numParticles_ == 1)
	{
		// If you have a single particle, the node has the same mass and CoM as it
		mass_ = particle_->getMass();
		centerOfMass_ = particle_->getPos();
	}
	else
	{
		mass_ = 0;
		centerOfMass_ = make_float2(0.0, 0.0);

		// If the node has more than one particle, it implies more than one child quadrant
		// Compute mass and CoM of all child quadrants
		for (int i = 0; i < quadrants.size(); i++)
		{
			if (!quadrants[i])
				continue;

			quadrants[i]->computeMassDistribution();
			const auto& childMass = quadrants[i]->getMass();
			const auto& childCoM = quadrants[i]->getCenterOfMass();
			mass_ += childMass;
			centerOfMass_ += childCoM * childMass;
		}
		// Divide the sum of all child CoM*mass by the total mass of the node
		centerOfMass_ /= mass_;
	}
}


bool BHQuadtreeNode::isRoot() const
{
	return parent_ == nullptr;
}

bool BHQuadtreeNode::isLeaf() const
{
	return quadrants[0] == nullptr &&
		quadrants[1] == nullptr &&
		quadrants[2] == nullptr &&
		quadrants[3] == nullptr;
}

BHQuadtreeNode::Quadrant BHQuadtreeNode::getQuadrant(const Particle * const particle)
{
	// Find which quadrant this particle belongs too
	const double x = particle->getPos().x;
	const double y = particle->getPos().y;
	Quadrant quadrant = Quadrant::NONE;
	if (x >= center_.x && y >= center_.y)
	{
		quadrant = Quadrant::NE;
	}
	else if (x <= center_.x && y >= center_.y)
	{
		quadrant = Quadrant::NW;
	}
	else if (x <= center_.x && y <= center_.y)
	{
		quadrant = Quadrant::SW;
	}
	else if (x >= center_.x && y <= center_.y)
	{
		quadrant = Quadrant::SE;
	}
	else
	{
		assert(false);
		// TODO: handle error
	}

	// Check if the quadrant exists, if not create a new one
	const auto quadIndex = static_cast<std::underlying_type_t<Quadrant>>(quadrant);
	if (!quadrants[quadIndex])
	{
		// Create the missing quadrant node
		auto count = ++nodeID_counter;
		switch (quadrant)
		{
		case Quadrant::NE: quadrants[quadIndex].reset(new BHQuadtreeNode(count, center_, max_, this)); break;
		case Quadrant::NW: quadrants[quadIndex].reset(new BHQuadtreeNode(count, make_float2(min_.x, center_.y),
			make_float2(center_.x, max_.y),
			this)); break;
		case Quadrant::SW: quadrants[quadIndex].reset(new BHQuadtreeNode(count, min_, center_, this)); break;
		case Quadrant::SE: quadrants[quadIndex].reset(new BHQuadtreeNode(count, make_float2(center_.x, min_.y),
			make_float2(max_.x, center_.y),
			this)); break;
		default: assert(false); //TODO: handle error
		}
	}

	return quadrant;
}

void BHQuadtreeNode::copyToArray(float* mass, int* child, float2* pos, int depth)
{
	mass[nodeID_ + NUM_PARTICLES] = mass_;
	pos[nodeID_ + NUM_PARTICLES] = centerOfMass_;
	for (int i = 0; i < 4; ++i)
	{
		if (!quadrants[i])
		{
			continue;
		}

		// If the quadrant isn't null, then copy its values to the node array
		if (quadrants[i]->getNumParticles() > 1)
		{
			child[4 * (nodeID_ + NUM_PARTICLES) + i] = quadrants[i]->getNodeID() + NUM_PARTICLES;
			quadrants[i]->copyToArray(mass, child, pos, depth+1);
		}
		else
		{
			child[4 * (nodeID_ + NUM_PARTICLES) + i] = quadrants[i]->getParticle()->getIdx();
		}
	}
}

/**
* Compute gravitational attraction of particle p2 on p1 and return p1's acceleration
*/
float2 BHQuadtreeNode::computeGravityAcc(const Particle * const p1, const Particle * const p2)
{
	float2 acc;

	float2 difference = p2->getPos() - p1->getPos();
	double dist = length(difference) + SOFTENER; // Distance between both points
	if (dist > 0)
	{
		const double g = GRAVITATIONAL_CONSTANT * p2->getMass() / (dist*dist*dist);
		acc.x = difference.x * g;
		acc.y = difference.y * g;
	}
	else
	{
		// If the two particles are in the same point, return 0
		acc.x = acc.y = 0;
	}

	return acc;
}

// Calculate acceleration on target_p due to force from this node's subtree on target_p
float2 BHQuadtreeNode::computeForceFromNode(const Particle * const target_p)
{
	float2 acc = make_float2(0.0f, 0.0f);

	if (numParticles_ == 1)
	{
		acc = computeGravityAcc(target_p, particle_);
	}
	else
	{
		float2 difference = centerOfMass_ - target_p->getPos();
		double dist = length(difference) + 0.001; // Distance between particle and node's CoM
		double nodeRadius = max_.x - min_.x;
		if (glm::abs(nodeRadius / dist) <= THETA)
		{
			// The particle is far enough to approximate the node as a single point
			const double g = GRAVITATIONAL_CONSTANT * mass_ / (dist*dist*dist);
			acc = difference * g;
		}
		else
		{
			// Recursively add acceleration due to child quadrants
			for (int i = 0; i < quadrants.size(); i++)
			{
				if (quadrants[i])
				{
					const auto accDueToQuad = quadrants[i]->computeForceFromNode(target_p);
					acc += accDueToQuad;
				}
			}
		}
	}

	return acc;
}
