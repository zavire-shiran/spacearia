#version 330

in vec2 texcoord;

out vec4 outputColor;

//vec4 gamma = vec4(1.0 / 2.2, 1.0/2.2, 1.0/2.2, 1.0);
vec4 gamma = vec4(1, 1, 1, 1);

uniform sampler2D tex;

void main()
{
  outputColor = texture(tex, texcoord);
  outputColor = pow(outputColor, gamma);
}