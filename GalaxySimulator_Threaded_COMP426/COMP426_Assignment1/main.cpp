
#include "libs\glew\glew.h"	// include GL Extension Wrangler
#include "libs\glfw\glfw3.h"	// include GLFW helper library
#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include "glm.hpp"
#include "gtc/matrix_transform.hpp"
#include "gtc/type_ptr.hpp"

#include "Shader.h"
#include "ParticleSystem.h"
using namespace std;

// Window dimensions
const GLuint WIDTH = 800, HEIGHT = 800;

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

	// Ask user for simulation parameters
	int numParticle = 9999;
	while (numParticle < 10 || numParticle > 5000)
	{
		cout << "Please enter the number of particles (10 to 5000) to simulate" << endl;
		cin >> numParticle;
	}

	// Load Shaders
	Shader shader("shaders/vertex.shader", "shaders/fragment.shader");
	shader.UseProgram();

	// Create particles and particle display system
	ParticleSystem particleSystem(numParticle);

	// Event loop
	int nbFrame = 0;
	float lastTime = glfwGetTime();
	while (!glfwWindowShouldClose(window))
	{
		// per-frame Time logic
		float currentFrame = glfwGetTime();
		nbFrame++;
		if (currentFrame - lastTime >= 1.0f)
		{
			printf("%f ms/frame or %f fps\n", 1000.0 / double(nbFrame), double(nbFrame));
			nbFrame = 0;
			lastTime += 1.0;
		}

		// Handle inputs
		processInput(window);

		// Render
		// Clear the colorbuffer
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Update the particles
		particleSystem.performComputations();

		//Draw
		shader.UseProgram();
		shader.setVec4("ColorIn", glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
		particleSystem.draw(0);
		shader.setVec4("ColorIn", glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
		particleSystem.draw(1);

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




