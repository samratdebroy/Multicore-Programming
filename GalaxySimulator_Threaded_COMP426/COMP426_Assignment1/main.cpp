
#include "libs\glew\glew.h"	// include GL Extension Wrangler
#include "libs\glfw\glfw3.h"	// include GLFW helper library
#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <random>
#include "glm.hpp"
#include "gtc/matrix_transform.hpp"
#include "gtc/type_ptr.hpp"

#include "Shader.h"
#include "ParticleDisplay.h"

using namespace std;

// Window dimensions
const GLuint WIDTH = 800, HEIGHT = 800;

// Timing Variables
float deltaTime = 0.0f; // Time b/w last frame and current frame
float lastFrame = 0.0f;

// Prototype
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

// The MAIN function, from here we start the application and run the game loop
int main()
{
	std::cout << "Starting GLFW context, OpenGL 4.4" << std::endl;
	// Init GLFW
	glfwInit();
	// Set all the required options for GLFW
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Create a GLFWwindow object that we can use for GLFW's functions
	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Window", nullptr, nullptr);
	if (window == nullptr)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Set the callback functions for frame size change
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions
	glewExperimental = GL_TRUE;
	// Initialize GLEW to setup the OpenGL Function pointers
	if (glewInit() != GLEW_OK)
	{
		std::cout << "Failed to initialize GLEW" << std::endl;
		return -1;
	}

	// Define the viewport dimensions
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	framebuffer_size_callback(window, width, height); // Sets the window size

	glEnable(GL_DEPTH_TEST); // enable the z-buffer and depth testing
	glDepthFunc(GL_LESS); // re-enable the depth buffer to normal


	// Ask user for simulation parameters
	int numParticle = 9999;
	double simTimeSeconds = 999;
	while (numParticle < 10 || numParticle > 5000)
	{
		cout << "Please enter the number of particles (10 to 5000) to simulate" << endl;
		cin >> numParticle;
	}

	while (simTimeSeconds < 10 || simTimeSeconds > 500)
	{
		cout << "Please enter the number of seconds (10 to 500) the simulation should run" << endl;
		cin >> simTimeSeconds;
	}

	Shader shader("shaders/vertex.shader", "shaders/fragment.shader");
	shader.UseProgram();

	// Random Number Generator
	std::mt19937 rng;
	rng.seed(std::random_device()());
	std::uniform_real_distribution<float> randFloat(-1.0f, 1.0f); // distribution in range [-1, 1]

	// Create particles and particle display system
	std::vector<Particle> particles;
	particles.reserve(numParticle);
	for (unsigned int i = 0; i < numParticle; i++)
	{
		particles.push_back(Particle(i, 1.0f));
		particles[i].setPos(glm::vec3(randFloat(rng), randFloat(rng), 0.0f));
	}
	ParticleDisplay particleDisplay(particles);

	// Game loop
	while (!glfwWindowShouldClose(window))
	{
		// per-frame Time logic
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		// Handle inputs
		processInput(window);

		// Render
		// Clear the colorbuffer
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// View and Model matrices
		glm::mat4 proj;
		glm::mat4 view;
		glm::mat4 model;

		shader.UseProgram();
		shader.setVec4("ColorIn", glm::vec4(1.0f, 1.0f, 0.0f, 1.0f));

		//Draw
		particleDisplay.draw();

		// Swap the screen buffers
		glfwSwapBuffers(window);
		// Check if any events have been activiated (key pressed, mouse moved etc.) and call corresponding response functions
		glfwPollEvents();
	}

	// Terminate GLFW, clearing any resources allocated by GLFW.
	glfwTerminate();
	return 0;
}

// Process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
void processInput(GLFWwindow *window)
{
	// Exits the application if escape was pressed
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions
	glViewport(0, 0, width, height);
}




