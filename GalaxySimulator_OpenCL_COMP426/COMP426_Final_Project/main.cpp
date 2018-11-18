
#include "glad/glad.h"	// include GL Extension manager
#include "glfw/glfw3.h"	// include GLFW helper library
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#include<atomic>
#include <mutex>
#include <condition_variable>
#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>

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
	glfwSwapInterval(0);
	// Set the callback functions for frame size change
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// Initialize GLAD to setup the OpenGL Function pointers
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	// Define the viewport dimensions
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	framebuffer_size_callback(window, width, height); // Sets the window size

	// Load Shaders
	Shader shader("shaders/vertex.shader", "shaders/fragment.shader");
	shader.UseProgram();

	// Create particles and particle display system
	auto glcontext = wglGetCurrentContext();
	auto HDC = wglGetCurrentDC();
	ParticleSystem particleSystem(glcontext, HDC);

	// Perform computations and update particles in a infinite loop in parallel to main thread
	std::mutex mutex;
	std::condition_variable cond;
	std::atomic_bool loopFlag = true;
	std::atomic_bool isDisplaying = false;

	auto performComputations = [&mutex, &cond, &particleSystem, &loopFlag, &isDisplaying]() {
		int nbFrame = 0;
		float lastPrintTime = glfwGetTime();
		while (loopFlag)
		{
			std::unique_lock<std::mutex> lock(mutex);
			while (isDisplaying) 
				cond.wait(lock);

			// Update FPS counter
			nbFrame++;
			float currentFrame = glfwGetTime();
			if (currentFrame - lastPrintTime >= 1.0f)
			{
				printf("Simulation Metrics: %f ms/frame or %f fps\n", 1000.0 / double(nbFrame), double(nbFrame));
				nbFrame = 0;
				lastPrintTime += 1.0;
			}
			// Update the particles
			particleSystem.performComputations();
			cond.notify_one();
			lock.unlock();
		}
	};
	std::thread computation_thread(performComputations);

	// Event loop
	float lastTime = glfwGetTime();
	constexpr float secPerFrame = 1.0f / 30;
	while (!glfwWindowShouldClose(window))
	{

		// per-frame Time logic
		float currentFrame = glfwGetTime();

		if (currentFrame - lastTime >= secPerFrame)
		{

			// Handle inputs
			processInput(window);

			// Render
			// Clear the colorbuffer
			glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			//Draw
			isDisplaying = true;
			std::unique_lock<std::mutex> lock(mutex);
			isDisplaying = false;
			shader.UseProgram();
			shader.setVec4("ColorIn", glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
			particleSystem.draw(0);
			shader.setVec4("ColorIn", glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
			particleSystem.draw(1);

			// Swap the screen buffers
			glfwSwapBuffers(window);
			// Check if any events have been activiated (key pressed, mouse moved etc.) and call corresponding response functions
			glfwPollEvents();
			lastTime += secPerFrame;
			//isDisplaying = false;
			cond.notify_one();
			lock.unlock();
		}
	}
	// Computation thread should stop
	loopFlag = false;
	computation_thread.join();
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




