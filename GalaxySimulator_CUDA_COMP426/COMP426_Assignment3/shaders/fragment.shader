#version 440 core

out vec4 FragColor;

uniform vec4 ColorIn = vec4(1.0, 0.0, 0.0, 1.0);

void main()
{
	FragColor = ColorIn;
}