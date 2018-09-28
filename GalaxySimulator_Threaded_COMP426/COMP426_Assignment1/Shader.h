#pragma once
#include <glew.h>
#include <string>
#include <fstream>
#include <iostream>
#include <mat4x2.hpp>

class Shader
{
public:
	// The program ID
	unsigned int ID;

	Shader(const std::string vertex_shader_path, const std::string fragment_shader_path);
	void UseProgram();

	// Helper functions to set uniforms
	void setMat4(const std::string &name, const glm::mat4 &mat) const;
	void setVec3(const std::string &name, const glm::vec3 &vec) const;
	void setVec4(const std::string &name, const glm::vec4 &vec) const;
};

