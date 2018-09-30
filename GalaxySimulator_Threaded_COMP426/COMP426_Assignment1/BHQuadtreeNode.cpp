#include "BHQuadtreeNode.h"
#include "SimulationConstants.h"

BHQuadtreeNode::BHQuadtreeNode(const glm::vec2 & min,
	const glm::vec2 & max,
	BHQuadtreeNode * parent)
	:centerOfMass_(0),
	min_(min),
	max_(max),
	center_(min.x + (max.x - min.x) * 0.5, min.y + (max.y - min.y) * 0.5),
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
		if (particle->getPos() == particle_->getPos())
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
		centerOfMass_ = glm::vec2(0.0, 0.0);

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
			centerOfMass_.x += childCoM.x * childMass;
			centerOfMass_.y += childCoM.y * childMass;
		}
		// Divide the sum of all child CoM*mass by the total mass of the node
		centerOfMass_.x /= mass_;
		centerOfMass_.y /= mass_;
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
		switch (quadrant)
		{
		case Quadrant::NE: quadrants[quadIndex].reset(new BHQuadtreeNode(center_, max_, this)); break;
		case Quadrant::NW: quadrants[quadIndex].reset(new BHQuadtreeNode(glm::vec2(min_.x, center_.y),
			glm::vec2(center_.x, max_.y),
			this)); break;
		case Quadrant::SW: quadrants[quadIndex].reset(new BHQuadtreeNode(min_, center_, this)); break;
		case Quadrant::SE: quadrants[quadIndex].reset(new BHQuadtreeNode(glm::vec2(center_.x, min_.y),
			glm::vec2(max_.x, center_.y),
			this)); break;
		default: assert(false); //TODO: handle error
		}
	}

	return quadrant;
}

/**
* Compute gravitational attraction of particle p2 on p1 and return p1's acceleration
*/
glm::vec2 BHQuadtreeNode::computeGravityAcc(const Particle * const p1, const Particle * const p2)
{
	glm::vec2 acc;

	glm::vec2 difference = p2->getPos() - p1->getPos();
	double dist = glm::length(difference) + SOFTENER; // Distance between both points
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
glm::vec2 BHQuadtreeNode::computeForceFromNode(const Particle * const target_p)
{
	glm::vec2 acc(0.0, 0.0);

	if (numParticles_ == 1)
	{
		acc = computeGravityAcc(target_p, particle_);
	}
	else
	{
		glm::vec2 difference = centerOfMass_ - target_p->getPos();
		double dist = glm::length(difference) + 0.001; // Distance between particle and node's CoM
		double nodeRadius = max_.x - min_.x;
		if (glm::abs(nodeRadius / dist) <= THETA)
		{
			// The particle is far enough to approximate the node as a single point
			const double g = GRAVITATIONAL_CONSTANT * mass_ / (dist*dist*dist);
			acc.x = difference.x * g;
			acc.y = difference.y * g;
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
