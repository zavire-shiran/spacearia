from pygame import display, OPENGL, DOUBLEBUF, FULLSCREEN, time
from OpenGL.GL import *
from OpenGL.GLU import *
import math

def init(size, fullscreen = False):
	global width, height, ratio
        width, height = size
	ratio = float(width)/float(height)
        flags = OPENGL | DOUBLEBUF
        if fullscreen:
                flags = flags | FULLSCREEN
        surface = display.set_mode(size, flags)

#        glEnable(GL_BLEND)
#        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

	glClearDepth(1)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glDisable(GL_CULL_FACE)

	glViewport(0, 0, width, height)

        return (ratio, 1.0)

def startframe():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

def endframe():
        display.flip()
