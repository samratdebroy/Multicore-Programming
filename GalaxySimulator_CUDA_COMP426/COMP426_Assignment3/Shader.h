#pragma once
#include <glad/glad.h>
#include <string>
#include <fstream>
#include <iostream>
#include <glm/mat4x2.hpp>

class Shader
{
public:
	// The program ID
	unsigned int ID;

	Shader(const std::string vertex_shader_path, const std::string fragment_shader_path);
	void UseProgram();

	// Helper functions to set uniforms
	void setVec4(const std::string &name, const glm::vec4 &vec) const;
};

