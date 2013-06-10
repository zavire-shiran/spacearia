from OpenGL.GL import *
import numpy
import texture
import math
from collections import defaultdict
import random
from ctypes import c_void_p
import pprint
import pygame
import time

currentworld = None
screensize = None

collision_epsillon = 0.001


def normalize(v, length = 1.0):
    l = math.sqrt(sum((x*x for x in v)))
    return tuple(x * length / l for x in v)


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

    def __del__(self):
        if glDeleteBuffers:
            glDeleteBuffers(1, [self.glbuffer])

    def addvertex(self, pos, texcoord):
        self.buffer += pos + texcoord
        self.numverts += 1

    def finalize_buffer(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.glbuffer)
#        print self.buffer
        glBufferData(GL_ARRAY_BUFFER, numpy.array(self.buffer, numpy.float32), GL_DYNAMIC_DRAW)

    def draw(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.glbuffer)

        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(self.pos_attrib_loc, self.possize, GL_FLOAT, GL_FALSE, (self.possize + self.texcoordsize) * 4, None)
        glVertexAttribPointer(self.texcoord_attrib_loc, self.texcoordsize, GL_FLOAT, GL_FALSE, (self.possize + self.texcoordsize) * 4, c_void_p(self.possize * 4))

        glDrawArrays(self.primtype, 0, self.numverts)


class World(object):
    def __init__(self, previous = None):
        pass

    def keydown(self, key):
        pass

    def keyup(self, key):
        pass

    def click(self, pos, button):
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

        texsamplers = ctypes.c_uint(0)
        glGenSamplers(1, texsamplers)
        self.texsampler = texsamplers.value
        glSamplerParameteri(self.texsampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glSamplerParameteri(self.texsampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glSamplerParameteri(self.texsampler, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glSamplerParameteri(self.texsampler, GL_TEXTURE_WRAP_T, GL_REPEAT)

        glClearColor(0.3, 0.3, 0.8, 0.0)

        self.camerapos = [40,25]
        self.scale = 0.15

        self.time = 0

        self.terrain = Terrain()
        self.playercontrols = {'left': False, 'right': False, 'jump': False}
        self.player = Player((40.0,25.0))
        self.gravity = -10
        self.terminalspeed = 15
        self.playermoveaccel = 10
        self.playermovespeed = 5

        self.projectiles = []

    def keydown(self, key):
        if key == pygame.K_a:
            self.playercontrols['left'] = True
        if key == pygame.K_d:
            self.playercontrols['right'] = True
        if key == pygame.K_SPACE:
            self.playercontrols['jump'] = True

    def keyup(self, key):
        if key == pygame.K_a:
            self.playercontrols['left'] = False
        if key == pygame.K_d:
            self.playercontrols['right'] = False
        if key == pygame.K_SPACE:
            self.playercontrols['jump'] = False

    def click(self, pos, button):
        screenratio = float(screensize[0]) / screensize[1]
        x, y = (((pos[0] * 6) - (3 * screenratio)) / self.scale + self.camerapos[0], 
                ((pos[1] * 6) - 3) / -self.scale + self.camerapos[1])
        tx, ty = math.floor(x), math.floor(y)
        if button == 1:
            pos = [self.player.pos[0] + self.player.size[0]/2,
                   self.player.pos[1] + self.player.size[1]/2]
            vel = normalize((x - pos[0], y - pos[1]), 30)
            self.projectiles.append(Projectile(pos, vel))
        elif button == 1 and not self.terrain.isfilled((tx, ty)) and (tx,ty) not in self.player.intersecting_tiles():
            self.terrain.addblock((tx, ty), 'dirt')
        elif button == 3 and self.terrain.isfilled((tx, ty)):
            self.terrain.removeblock((tx, ty))
        

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
        
        self.terrain.draw()
        self.player.draw()
        for proj in self.projectiles:
            proj.draw()

    def collide_with_terrain(self, actor):
        actor.canjump = False

        # collision detect/handling w/ terrain
        for tile in actor.intersecting_tiles():
            # these checks need to be in a better order
            # the tile most directly under should be first, and diagonals should be last
            if self.terrain.isfilled(tile):
                tleft, tbottom = tile
                tright = tleft + 1
                ttop = tbottom + 1

                pleft, pbottom = actor.pos
                pright = actor.size[0] + pleft
                ptop = actor.size[1] + pbottom

                boverlap = ttop - pbottom
                toverlap = ptop - tbottom
                loverlap = tright - pleft
                roverlap = pright - tleft

                if min(loverlap, roverlap) < collision_epsillon or min(boverlap, toverlap) < collision_epsillon:
                    continue
                
                if min(boverlap, toverlap) < min(loverlap, roverlap):
                    #vertical collision
                    if boverlap < toverlap:
                        #tile is under actor
                        actor.pos[1] = float(ttop)
                        actor.canjump = True
                        if actor.velocity[1] < 0.0:
                            actor.velocity[1] = 0.0
                    else:
                        #tile is over actor
                        actor.pos[1] = float(tbottom) - actor.size[1]
                        if actor.velocity[1] > 0.0:
                            actor.velocity[1] = 0.0
                elif min(loverlap, roverlap) > collision_epsillon:
                    if loverlap < roverlap:
                        #tile is left of actor
                        actor.pos[0] = float(tright)
                        if actor.velocity[0] < 0.0:
                            actor.velocity[0] = 0.0
                    else:
                        #tile is right of actor
                        actor.pos[0] = float(tleft) - actor.size[0]
                        if actor.velocity[0] > 0.0:
                            actor.velocity[0] = 0.0

    def step(self, dt):
        # kinematics update
        if self.playercontrols['left']:
            self.player.velocity[0] -= self.playermoveaccel * dt
        if self.playercontrols['right']:
            self.player.velocity[0] += self.playermoveaccel * dt
        if abs(self.player.velocity[0]) > self.playermovespeed:
            self.player.velocity[0] = math.copysign(self.playermovespeed, self.player.velocity[0])
        self.player.velocity[1] += self.gravity * dt

        if abs(self.player.velocity[1]) > self.terminalspeed:
            self.player.velocity[1] = math.copysign(self.terminalspeed, self.player.velocity[1])
        self.player.pos[0] += self.player.velocity[0] * dt
        self.player.pos[1] += self.player.velocity[1] * dt

        if self.player.canjump and self.playercontrols['jump']:
            self.player.velocity[1] = 10
        if self.player.canjump and not self.playercontrols['left'] and not self.playercontrols['right']:
            if self.player.velocity[0] != 0:
                accel = -math.copysign(self.playermoveaccel * dt, self.player.velocity[0])
                if abs(self.player.velocity[0]) < accel:
                    self.player.velocity[0] = 0.0
                else:
                    self.player.velocity[0] += accel

        self.collide_with_terrain(self.player)

        self.camerapos = list(self.player.center)

        # update rendering
        self.player.generate_prims()

        for proj in self.projectiles:
            proj.step(dt)

        i = 0
        while i < len(self.projectiles):
            x, y = self.projectiles[i].pos
            tx, ty = math.floor(x), math.floor(y)
            if self.projectiles[i].shoulddie() or self.terrain.isfilled((tx,ty)):
                del self.projectiles[i]
            else:
                i += 1


class Player(object):
    def __init__(self, pos):
        self.pos = list(pos)
        self.velocity = [0.0, 0.0]
        self.size = (1.2, 2.5)
        self.canjump = False
        self.generate_prims()

    @property
    def center(self):
        return (self.pos[0] + self.size[0]/2, self.pos[1] + self.size[1]/2)

    def generate_prims(self):
        x, y = self.pos
        w, h = self.size
        self.prims = Primitives(GL_QUADS, 0, 1)
        self.prims.addvertex((x, y), (0.5, 0.5))
        self.prims.addvertex((x+w,y), (1.0, 0.5))
        self.prims.addvertex((x+w, y+h), (1.0, 1.0))
        self.prims.addvertex((x, y+h), (0.5, 1.0))
        self.prims.finalize_buffer()
    
    def intersecting_tiles(self):
        x, y = self.pos
        w, h = self.size
        xstart = int(math.floor(x))
        xend = int(math.ceil(x + w))
        ystart = int(math.floor(y))
        yend = int(math.ceil(y+h))
        return ((x, y) for x in xrange(xstart, xend) for y in xrange(ystart, yend))

    def draw(self):
        self.prims.draw()


class Projectile(object):
    def __init__(self, pos, vel):
        self.pos = list(pos)
        self.vel = list(vel)
        self.generate_prims()
        self.maxlife = 2.0
        self.lived = 0.0

    def step(self, dt):
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt
        self.lived += dt
        self.generate_prims()

    def shoulddie(self):
        return self.lived >= self.maxlife

    def generate_prims(self):
        x, y = self.pos
        vx, vy = self.vel
        px, py = normalize((-vy, vx), 0.1) # perpendicular to direction of travel
        tailx, taily = normalize((-vx, -vy), 3.0)
        self.prims = Primitives(GL_QUADS, 0, 1)
        self.prims.addvertex((x - px, y - py), (0.5, 0.0))
        self.prims.addvertex((x + px, y + py), (0.5, 0.5))
        self.prims.addvertex((x + px + tailx, y + py + taily), (1.0, 0.5))
        self.prims.addvertex((x - px + tailx, y - py + taily), (1.0, 0.0))
        self.prims.finalize_buffer()

    def draw(self):
        self.prims.draw()


class Terrain(object):
    def __init__(self):
        self.prims = None
        self.width = 80
        self.height = 20
        self.tiles = {}
        for x in xrange(self.width):
            for y in xrange(self.height):
                if random.random() > 0.8:
                    kind = 'rock'
                else:
                    kind = 'dirt'
                self.tiles[(x,y)] = kind
        for y in xrange(self.height, self.height + 10):
            self.tiles[(10, y)] = 'dirt'
            self.tiles[(70, y)] = 'dirt'
        self.generate_prims()

    def draw(self):
        self.prims.draw()

    def isfilled(self, pos):
        return tuple(pos) in self.tiles

    def addblock(self, pos, kind):
        if pos not in self.tiles:
            self.tiles[pos] = kind
            self.generate_prims()

    def removeblock(self, pos):
        if pos in self.tiles:
            kind = self.tiles[pos]
            del self.tiles[pos]
            self.generate_prims()
            return kind
        return None

    def generate_prims(self):
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
