#version 330

uniform vec2 CameraCenter;
uniform mat4 CameraToClipTransform;
uniform float scale;

layout(location=0) in vec2 position;
layout(location=1) in vec2 in_texcoord;

smooth out vec2 texcoord;

void main()
{
  vec4 pos = vec4((position - CameraCenter) * vec2(scale, scale), 0.0, 1.0);
  gl_Position = CameraToClipTransform * pos;
  texcoord = in_texcoord;
}
