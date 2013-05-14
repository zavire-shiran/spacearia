from OpenGL.GL import *
import numpy
import texture
import math
from collections import defaultdict
import random
from ctypes import c_void_p
import pprint
import pygame

currentworld = None
screensize = None


def setscreensize(size):
    global screensize
    screensize = size


def getworld():
    global currentworld
    return currentworld


def transitionto(world):
    global currentworld
    currentworld = world(currentworld)


def createshader(filename, shadertype):
    shader = glCreateShader(shadertype)
    source = open(filename).read()
    glShaderSource(shader, source)
    glCompileShader(shader)
    ok = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not ok:
        print 'Shader compile failed:', filename
        print glGetShaderInfoLog(shader)
        # this should probably be an exception, but I can't be arsed right now.
        return -1
    return shader


def createprogram(*shaders):
    program = glCreateProgram()
    for shader in shaders:
        glAttachShader(program, shader)
    glLinkProgram(program)
    ok = glGetProgramiv(program, GL_LINK_STATUS)
    if not ok:
        print 'Could not link program:'
        print glGetProgramInfoLog(program)
        return -1
    return program


def make_ortho_matrix(left, right, bottom, top, near, far):
    return numpy.array([[2 / float(right - left), 0, 0, -float(right + left) / (right - left)],
                        [0, 2 / float(top - bottom), 0, -float(top + bottom) / (top - bottom)],
                        [0, 0, 2 / float(far - near), -float(far + near) / (far - near)],
                        [0, 0, 0, 1]], numpy.float32)


class Primitives:
    def __init__(self, primtype, pos_attrib_loc, texcoord_attrib_loc):
        self.buffer = []
        self.glbuffer = glGenBuffers(1)
        self.primtype = primtype
        self.pos_attrib_loc = pos_attrib_loc
        self.texcoord_attrib_loc = texcoord_attrib_loc
        self.numverts = 0
        self.possize = 2
        self.texcoordsize = 2

    def addvertex(self, pos, texcoord):
        self.buffer += pos + texcoord
        self.numverts += 1

    def finalize_buffer(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.glbuffer)
#        print self.buffer
        glBufferData(GL_ARRAY_BUFFER, numpy.array(self.buffer, numpy.float32), GL_STATIC_DRAW)

    def draw(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.glbuffer)

        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(self.pos_attrib_loc, self.possize, GL_FLOAT, GL_FALSE, (self.possize + self.texcoordsize) * 4, None)
        glVertexAttribPointer(self.texcoord_attrib_loc, self.texcoordsize, GL_FLOAT, GL_FALSE, (self.possize + self.texcoordsize) * 4, c_void_p(self.possize * 4))

        glDrawArrays(self.primtype, 0, self.numverts)


class World:
    def __init__(self, previous = None):
        pass

    def keydown(self, key):
        pass

    def keyup(self, key):
        pass

    def click(self, pos):
        pass

    def draw(self):
        pass

    def step(self, dt):
        pass


class Game(World):
    def __init__(self, previous = None):
        vertshader = createshader('color_vertex.shader', GL_VERTEX_SHADER)
        fragshader = createshader('color_fragment.shader', GL_FRAGMENT_SHADER)
        self.shaderprogram = createprogram(vertshader, fragshader)

        self.camera_center_uniform = glGetUniformLocation(self.shaderprogram, 'CameraCenter')
        self.camera_to_clip_uniform = glGetUniformLocation(self.shaderprogram, 'CameraToClipTransform')
        self.texture_uniform = glGetUniformLocation(self.shaderprogram, 'tex')
        self.scale_uniform = glGetUniformLocation(self.shaderprogram, 'scale')

        self.tex = texture.Texture('terrain.png')

        self.tris = Primitives(GL_TRIANGLES, 0, 1)
        self.tris.addvertex((0, 0), (0, 0))
        self.tris.addvertex((0, 1), (0, 1))
        self.tris.addvertex((1, 1), (1, 1))
        self.tris.finalize_buffer()

        texsamplers = ctypes.c_uint(0)
        glGenSamplers(1, texsamplers)
        self.texsampler = texsamplers.value
        glSamplerParameteri(self.texsampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glSamplerParameteri(self.texsampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glSamplerParameteri(self.texsampler, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glSamplerParameteri(self.texsampler, GL_TEXTURE_WRAP_T, GL_REPEAT)

        self.camerapos = [10,5]
        self.scale = 0.3

        self.time = 0

        self.camcontrols = {'left': False, 'right': False, 'up': False, 'down': False, 'zoomin':False, 'zoomout':False}
        self.terrain = Terrain()

    def keydown(self, key):
        pass

    def keyup(self, key):
        pass

    def draw(self):
        glUseProgram(self.shaderprogram)

        screenratio = float(screensize[0]) / screensize[1]

        glUniform2fv(self.camera_center_uniform, 1, self.camerapos)
        glUniformMatrix4fv(self.camera_to_clip_uniform, 1, False, make_ortho_matrix(-3 * screenratio, 3 * screenratio, -3, 3, 5, -5))
        glUniform1f(self.scale_uniform, self.scale)

        glBindSampler(1, self.texsampler)

        glActiveTexture(GL_TEXTURE0 + 1)
        self.tex()
        glUniform1i(self.texture_uniform, 1)
        
        self.tris.draw()
        self.terrain.draw()

    def step(self, dt):
        pass


class Terrain:
    def __init__(self):
        self.prims = None
        self.width = 20
        self.height = 10
        self.tiles = {}
        for x in xrange(self.width):
            for y in xrange(self.height):
                if random.random() > 0.5:
                    kind = 'rock'
                else:
                    kind = 'dirt'
                self.tiles[(x,y)] = kind
        self.recalc_prims()
    def draw(self):
        self.prims.draw()
    def recalc_prims(self):
        self.prims = Primitives(GL_QUADS, 0, 1)
        for pos, kind in self.tiles.items():
            x, y = pos
            poscoords = [(x, y), (x+1, y), (x+1,y+1), (x,y+1)]
            if kind == 'rock':
                texcoords = [(0, 0.5), (0, 1), (0.5, 1), (0.5, 0.5)]
            else:
                texcoords = [(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0)]
            for pos, tex in zip(poscoords, texcoords):
                self.prims.addvertex(pos, tex)
        self.prims.finalize_buffer()
