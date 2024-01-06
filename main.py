import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr

class StaticComponent:

    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)

class DynamicComponent:

    def __init__(self, position, eulers, health):
        self.position = np.array(position, dtype = np.float32)
        self.eulers = np.array(eulers, dtype = np.float32)
        self.velocity = np.array([0,0,0], dtype = np.float32)
        self.state = "stable"
        self.health = health
        self.canShoot = True
        self.reloading = False
        self.reloadTime = 0

class Scene:

    def __init__(self, objList):

        self.objList = objList
        self.objectFilenameList = []
        self.meshes = []
        self.staticComponents = []
        self.dynamicComponents = []
        self.bullets = []
        self.powerups = []
        self.enemySpawnRate = 0.1
        self.powerupSpawnRate = 0.05
        self.enemyShootRate = 0.1

        for obj in self.objList:
            self.objectFilenameList.append(obj.filename)
            if obj.isStatic == 1:
                self.staticComponents.append(StaticComponent(obj.position, obj.eulers))
            else:
                self.dynamicComponents.append(DynamicComponent(obj.position, obj.eulers, obj.health))
            self.meshes.append(MeshNoTex("objects/"+obj.filename, obj.position))


        self.player = DynamicComponent(
            position = [0,0,0],
            eulers = [0, 90, 0],
            health = 30
        )
        

        

    def update(self, rate):
        pass

    def move_player(self,dPos):
        pass


class GraphicsEngine:

    def __init__(self):
        
        self.colorPalette = {
            "darkBlue": np.array([0,13/255,107/255], dtype = np.float32),
            "yellow": np.array([246/255,236/255,169/255], dtype = np.float32)
        }

        # Initializing GE with pygame and opengl:
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK,
                                    pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode((640,480), pg.OPENGL|pg.DOUBLEBUF)

        glClearColor(self.colorPalette["darkBlue"][0],self.colorPalette["darkBlue"][1],self.colorPalette["darkBlue"][2], 1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)

        shader = self.createShader("shaders/vertex","shaders/fragment")
        self.renderPass = RenderPass(shader)

    def createShader(self, vertexFilepath, fragmentFilepath):

        with open(vertexFilepath, 'r') as f:
            vertexSrc = f.readlines()

        with open(fragmentFilepath, 'r') as f:
            fragmentSrc = f.readlines()
        shader = compileProgram(compileShader(vertexSrc, GL_VERTEX_SHADER),
                                compileShader(fragmentSrc, GL_FRAGMENT_SHADER))
        return shader
    def render(self,scene):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.renderPass.render(scene, self)
        pg.display.flip()

    def destroy(self):
        pg.quit()


class CameraControl:
    def __init__(self):
        self.camera_position = np.array([0, 0, 0], dtype=np.float32)
        self.camera_rotation = np.array([0, 0, 0], dtype=np.float32)
        self.camera_speed = 0.01
        self.last_mouse_position = np.array(pg.mouse.get_pos(), dtype=np.float32)
        self.camera_target = np.array([0, 0, 0], dtype=np.float32)
        self.camera_up = np.array([0, 1, 0], dtype=np.float32)
        self.i = 0
    def applyVector(self, vector):
        self.camera_position += vector

    def euler_to_quaternion(self):
        pitch = np.radians(self.camera_rotation[0])
        yaw = np.radians(self.camera_rotation[1])

        qx = np.sin(pitch/2) * np.cos(yaw/2)
        qy = np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(pitch/2) * np.cos(yaw/2) - np.sin(pitch/2) * np.sin(yaw/2)
        qw = np.sin(pitch/2) * np.sin(yaw/2) + np.cos(pitch/2) * np.cos(yaw/2)

        return np.array([qx, qy, qz, qw], dtype=np.float32)

    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return np.array([w, x, y, z], dtype=np.float32)

    def rotate_vector(self, vector):
        quaternion = self.euler_to_quaternion()
        q_vector = np.array([0, *vector], dtype=np.float32)

        # Quaternion rotation formula: q * v * q^-1
        q_conjugate = np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]], dtype=np.float32)
        rotated_vector = self.quaternion_multiply(self.quaternion_multiply(quaternion, q_vector), q_conjugate)
        return rotated_vector[1:]  # Ignore the w component

    def handleKeys(self):
        keys = pg.key.get_pressed()
        direction = np.array([0, 0, 0], dtype=np.float32)
        if keys[pg.K_d]:
            direction += np.array([0, 0, self.camera_speed], dtype=np.float32)  # Move right
        if keys[pg.K_a]:
            direction += np.array([0, 0, -self.camera_speed], dtype=np.float32)  # Move left
        if keys[pg.K_s]:
            direction += np.array([-self.camera_speed, 0, 0], dtype=np.float32)  # Move backwards
        if keys[pg.K_w]:
            direction += np.array([self.camera_speed, 0, 0], dtype=np.float32)  # Move forwards
        if keys[pg.K_SPACE]:
            direction += np.array([0, self.camera_speed, 0], dtype=np.float32)  # Move up
        if keys[pg.K_LSHIFT]:
            direction += np.array([0, -self.camera_speed, 0], dtype=np.float32)  # Move down

        # Calculate the movement direction based on the camera's rotation
        pitch = np.radians(self.camera_rotation[0])
        yaw = np.radians(self.camera_rotation[1])
        direction = np.array([
            direction[0]*np.cos(yaw) - direction[2]*np.sin(yaw),
            direction[1],
            direction[0]*np.sin(yaw) + direction[2]*np.cos(yaw)
        ], dtype=np.float32)

        self.applyVector(direction)
        print(self.camera_position)
            

    def handleMouse(self):

        width, height = pg.display.get_surface().get_size()
        
        center = [width / 2, height / 2]

        mouse_position = np.array(pg.mouse.get_pos(), dtype=np.float32)
        mouse_movement = mouse_position - center
        #mouse_movement /= 100  # Scale down the mouse movement

        self.camera_rotation[0] -= mouse_movement[1]  # pitch
        self.camera_rotation[1] += mouse_movement[0]  # yaw

        self.camera_rotation[0] = np.clip(self.camera_rotation[0], -89, 89)  # Clamp pitch
        self.camera_rotation[1] %= 360  # Wrap yaw

        # Calculate the new target position
        if np.abs(self.camera_rotation[0]) == 90:
            direction = np.array([0, np.sign(self.camera_rotation[0]), 0], dtype=np.float32)
        else:
            direction = np.array([
                np.cos(np.radians(self.camera_rotation[1])) * np.cos(np.radians(self.camera_rotation[0])),
                np.sin(np.radians(self.camera_rotation[0])),
                np.sin(np.radians(self.camera_rotation[1])) * np.cos(np.radians(self.camera_rotation[0]))
            ], dtype=np.float32)
        self.camera_target = self.camera_position + direction
        print(self.camera_rotation)
        self.i += 1
        if self.i == 2:
            pg.mouse.set_pos(center)
            self.i = 0  # Reset mouse position
        pg.mouse.set_visible(False) 

class RenderPass:

    def __init__(self, shader):
        self.shader = shader
        self.cameraControl = CameraControl()
        glUseProgram(self.shader)
        projectionTransform = pyrr.matrix44.create_perspective_projection(
            fovy=90, aspect = 800/600,
            near = 0.1, far = 100, dtype=np.float32
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "projection"),
            1, GL_FALSE, projectionTransform
        )
        self.modelMatrixLocation = glGetUniformLocation(self.shader, "model")
        self.viewMatrixLocation = glGetUniformLocation(self.shader, "view")
        self.colorLoc = glGetUniformLocation(self.shader, "object_color")

    def render(self, scene, engine):
        glUseProgram(self.shader)

        viewTransform = pyrr.matrix44.create_look_at(
            eye = self.cameraControl.camera_position,
            target = np.array(self.cameraControl.camera_target, dtype = np.float32),
            up = np.array([0,1,0], dtype = np.float32), dtype = np.float32
        )
        glUniformMatrix4fv(self.viewMatrixLocation, 1, GL_FALSE, viewTransform)

        for static in scene.meshes:
            glUniform3fv(self.colorLoc, 1, engine.colorPalette["yellow"])
            modelTransform = pyrr.matrix44.create_identity(dtype=np.float32)

            modelTransform = pyrr.matrix44.multiply(
                modelTransform,
                pyrr.matrix44.create_from_eulers(np.array([0,0,0], dtype=np.float32))
            )
            modelTransform = pyrr.matrix44.multiply(
                m1 = modelTransform,
                m2 = pyrr.matrix44.create_from_translation(vec = np.array(static.position, dtype = np.float32))
            )
            glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, modelTransform)

            glBindVertexArray(static.vao)
            glDrawArrays(GL_TRIANGLES, 0, static.vertexCount)

    def destroy(self):
        glDeleteProgram(self.shader)


class MeshNoTex:

    def __init__(self, filename, position):
        self.position = position
        self.verticies = self.loadMesh(filename)
        self.vertexCount = len(self.verticies)//3
        self.verticies = np.array(self.verticies, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER,self.verticies.nbytes, self.verticies, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

    def loadMesh(self,filename):
        v = []
        verticies = []

        with open(filename, 'r')as f:
            line = f.readline()
            while line:
                firstSpace = line.find(" ")
                flag = line[0:firstSpace]
                if flag == "v":
                    line = line.replace("v ","")
                    line = line.split(" ")
                    l=[float(x) for x in line]
                    v.append(l)
                elif flag == "f":
                    line = line.replace("f ","")
                    line = line.replace("\n ","")
                    line = line.split(" ")
                    faceVerticies = []
                    for vertex in line:
                        l = vertex.split("/")
                        position = int(l[0])
                        faceVerticies.append(v[position-1])
                    trianglesInFace = len(line) - 2

                    vertexOrder = []

                    for i in range(trianglesInFace):
                        vertexOrder.append(0)
                        vertexOrder.append(i+1)
                        vertexOrder.append(i+2)
                    for i in vertexOrder:
                        for x in faceVerticies[i]:
                            verticies.append(x)
                line = f.readline()
        return verticies
    
    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))


class App:

    def __init__(self, screenWidth, screenHeight):

        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        self.objectFilenameList = self.getObjectList()
        self.renderer = GraphicsEngine()
        self.scene = Scene(self.objectFilenameList)
        self.lastTime = pg.time.get_ticks()
        self.currentTime = 0
        self.numFrames = 0
        self.frameTime = 0
        self.lightCount = 0

        self.mainloop()

    def mainloop(self):
        run = True
        while (run):

            for event in pg.event.get():
                if (event.type == pg.QUIT):
                    run = False

            

            self.scene.update(self.frameTime*0.05)

            self.renderer.render(self.scene)
            self.renderer.renderPass.cameraControl.handleKeys()
            self.renderer.renderPass.cameraControl.handleMouse()
            self.calculateFramerate()
        self.quit()


    def calculateFramerate(self):

        self.currentTime = pg.time.get_ticks()
        delta = self.currentTime - self.lastTime

        if (delta >= 1000):
            framerate = max(1,int(1000.0 * self.numFrames/delta))
            pg.display.set_caption(f"{framerate} fps")
            self.lastTime = self.currentTime
            self.numFrames = -1
            self.frameTime = float(1000.0 / max(1,framerate))
        self.numFrames += 1

    def getObjectList(self):
        objList = []
        with open("objects/objectList", "r") as f:
            line = f.readline()
            while line:
                words = line.split(" ")
                objList.append(RawObject(words[0], [float(words[1]), float(words[2]), float(words[3])], int(words[4])))
                line = f.readline()
        return objList
    def quit(self):
        self.renderer.destroy()


class RawObject:
    def __init__(self, filename, position, isStatic):
        self.filename = filename
        self.position = position
        self.isStatic = isStatic
        self.eulers = [0,0,0]
        self.health = 10

        
    
myApp = App(800,600)