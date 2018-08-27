import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
import pybullet as p
import pybullet_data

import random
import math
import numpy as np
import time

#from . import walkGenerator
#from . import motorController

import time
from time import sleep


class BipedalRobotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        # 물리 값 설정
        self.gravity = -9.8
        self.timeStep = 1.0/1000
        self.numSolverIterations = 200
        self.friction = 0.8
        self.spinFrictino = 0.8

        self.footfriction = 0.8
        self.perturbation = 0.05

        self.path_running = os.path.abspath(os.path.dirname(__file__))
        
        isDirectConnect = False
        # 물리 엔진 셋업
        if isDirectConnect == True:
            self.physicsClient = p.connect(p.DIRECT)
        else:
            self.physicsClient = p.connect(p.GUI)
        p.setTimeStep(self.timeStep)
        p.setPhysicsEngineParameter(numSolverIterations=self.numSolverIterations)
        p.setGravity(0, 0, self.gravity)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane = p.loadURDF("plane.urdf")

        # 모터 밸류
        self.motor_kp = 0.5
        self.motor_kd = 0.5
        self.motor_torque = 1.5
        self.motor_speed = 5.5
        self.motor_friction = 0.0

        # 학습 환경 설정
        self.action_space = spaces.Box(np.array([-1, -1, -1, -1, -1]),
                                       np.array([+1, +1, +1, +1, +1]))
        # # 넘어짐 판정
        self.fall_criteria = 0.7
        # # 지형 설정
        self.setLoadTerrain = True
        self.loadTerrain_flat = [True]  # 0
        self.loadTerrain_weak_noise = [False]  # 1
        self.loadTerrain_strong_noise = [True]  # 2
        self.loadTerrain_square = [False]  # 3
        self.terrainSwitch = 0
        # # 공던지기 설정
        self.ballPath = os.path.abspath(os.path.dirname(__file__))+ '/resources'+'/ball.xml'
        self.ball = p.loadURDF(self.ballPath,[0,20,0.5], useFixedBase=False) 
        self.throwBallEnable = False
        self.throwBallEnableList = [False, False, False, False]
        self.throwBallForce = 0.5/self.timeStep
        self.throwBallDistance = 0.5

        self._observation = []
        # 관찰 공간                          롤, 피치,  x차이, z플러스,R leg 롤+, R leg 피치+ , L leg 롤+, L leg 피치+ , RR torque, RP torque, LR torque, LP torque
        self.observation_space = spaces.Box(np.array([-1, -1, -150, -50, -3, -3, -3, -3, -1.5, -1.5, -1.5, -1.5, ]),
                                            np.array([+1, +1, +150, +50, +3, +3, +3, +3, +1.5, +1.5, +1.5, +1.5, ]))
        #                                             롤,피치, x dif,z dif,RRa,
        # 리소스 설정
        self.urdf_robot_path = os.path.abspath(os.path.dirname(__file__))+ '/resources'+'/humanoid_leg_12dof.7.urdf'

        self.fall_down = False
        # 보행 중 변수 설정
        self.walkStepCounter = 0
        self.walkRightStep = True  # 지금이 오른발거름 해야될때는 True. 오른 발걸음 끝나고 나서는 False
        # 로봇과 보행 파라미터 설정
        self.pos_z_plus = 0
        '''
        urdf_loaded = False
        while urdf_loaded == False:
            try:
                self.robot = p.loadURDF(self.urdf_robot_path, useFixedBase=False)
                urdf_loaded = True
            except:
                time.sleep(0.5)
        '''
        self.walk = WalkGenerator()
        self.walk.setWalkParameter(bodyMovePoint=8, legMovePoint=8, h=50, l=90, sit=50, swayBody=45, swayFoot=0,
                                   bodyPositionXPlus=5, swayShift=3, weightStart=0.5, weightEnd=0.7, stepTime=0.06, damping=0.0, incline=0.0)
        self.walk.generate()
        # self.walk.showGaitPoint3D()
        # self.walk.inverseKinematicsAll()

        #self.terrain = p.loadURDF(os.path.join(path, "ball.xml"), [0, 10, 0.2])
        self.rightFoot_roll_state = []
        self.rightFoot_pitch_state = []
        self.leftFoot_roll_state = []
        self.leftFoot_pitch_state = []

        self.floatingFoot_roll_state_observe = []
        self.floatingFoot_pitch_state_observe = []
        self.groundFoot_roll_state_observe = []
        self.groundFoot_pitch_state_observe = []

        self.floatingFoot_roll_torque = []
        self.floatingFoot_pitch_torque = []
        self.groundFoot_roll_torque = []
        self.groundFoot_pitch_torque = []

        self.terrainYpos = []

        self.reward = []

        self.testCount = []

        self.envNumStr = ''

        self.seed()

        self.worker_count = 1

    def setEnvironmentNumber(self, num):
        self.envNumStr = str(num)
        self.urdf_robot_path = os.path.abspath(os.path.dirname(__file__))+ '/resources' + self.envNumStr + '/humanoid_leg_12dof.7.urdf'
        self.ballPath = os.path.abspath(os.path.dirname(__file__))+ '/resources' + self.envNumStr + '/ball.xml'
        self.robot = p.loadURDF(self.urdf_robot_path,[0,0,1], useFixedBase=False)
        self.motors = MotorController(
            self.robot, self.motor_kp, self.motor_kd, self.motor_torque, self.motor_speed, self.timeStep)

    def makeWorld(self, worker_count):
        p.resetSimulation()
        p.setTimeStep(self.timeStep)
        p.setPhysicsEngineParameter(numSolverIterations=self.numSolverIterations)
        p.setGravity(0, 0, self.gravity)
        plane = p.loadURDF("plane100.urdf")

        self.rightFoot_roll_state = np.zeros((worker_count), dtype='f')
        self.rightFoot_pitch_state = np.zeros((worker_count), dtype='f')
        self.leftFoot_roll_state = np.zeros((worker_count), dtype='f')
        self.leftFoot_pitch_state = np.zeros((worker_count), dtype='f')

        self.floatingFoot_roll_state_observe = np.zeros((worker_count), dtype='f')
        self.floatingFoot_pitch_state_observe = np.zeros((worker_count), dtype='f')
        self.groundFoot_roll_state_observe = np.zeros((worker_count), dtype='f')
        self.groundFoot_pitch_state_observe = np.zeros((worker_count), dtype='f')

        self.floatingFoot_roll_torque = np.zeros((worker_count), dtype='f')
        self.floatingFoot_pitch_torque = np.zeros((worker_count), dtype='f')
        self.groundFoot_roll_torque = np.zeros((worker_count), dtype='f')
        self.groundFoot_pitch_torque = np.zeros((worker_count), dtype='f')

        self.terrainYpos = np.zeros((worker_count), dtype='f')

        self.reward = np.zeros((worker_count), dtype='f')

        self.envStepCounter = np.zeros((worker_count), dtype='i')
        self.walkStepCounter = np.zeros((worker_count), dtype='i')
        self.pos_z_plus = np.zeros((worker_count), dtype='f')
        self.fall_down = np.zeros((worker_count), dtype='b')

        self.robot = []
        self.motors = []
        self.initial_position = np.array([[0,0,0], [0,13,0], [0,26,0], [0,39,0], [13,0,0], [13,13,0], [13,26,0], [13,39,0]])

        for i in range(worker_count):
            self.robot[i] = p.loadURDF(self.urdf_robot_path, self.initial_position[i]+[0, 0, 0.2], useFixedBase=False)
            self.motors[i] = MotorController(self.robot[i], self.motor_kp, self.motor_kd, self.motor_torque, self.motor_speed, self.timeStep)


        if self.setLoadTerrain:
            p.resetSimulation()
            p.setTimeStep(self.timeStep)
            p.setPhysicsEngineParameter(numSolverIterations=self.numSolverIterations)
            self.robot = p.loadURDF(self.urdf_robot_path, [0, 0, 0.2], useFixedBase=False)
            self.motors.setRobot(self.robot)
            #self.ball = p.loadURDF(os.path.join(os.path.abspath(os.path.dirname(__file__)), "ball.xml"), [0, 15, 0.01])
            #plane = p.loadURDF("plane.urdf")
            self.loadTerrain()
        '''
        p.resetSimulation()
        p.setTimeStep(self.timeStep)
        p.setPhysicsEngineParameter(numSolverIterations=self.numSolverIterations)
        self.robot = p.loadURDF(self.urdf_path, useFixedBase=False)
        self.motors.setRobot(self.robot)
        if(self.throwBallEnable == True):
            self.ball = p.loadURDF(os.path.join(os.path.abspath(os.path.dirname(__file__)), "ball.xml"), [0, 10, 1])
        p.setGravity(0, 0, 0)
        self.inverseKinematicsMove(
            self.walk._walkPointLeftStepRightLeg[:, self.walk._stepPoint-4], self.walk._walkPointLeftStepLeftLeg[:, self.walk._stepPoint-4], 0, 0.3, 0.2)
        '''
        orientation = p.getQuaternionFromEuler([0, 0, 0])

        p.setGravity(0, 0, 0)
        #p.resetBasePositionAndOrientation(self.robot, [0, 0, 0.2], orientation)
        self.inverseKinematicsMove(
            self.walk._walkPointStartLeftstepRightLeg[:, 0], self.walk._walkPointStartLeftstepLeftLeg[:, 0], 0, 0.2, 0.3)

        p.resetBasePositionAndOrientation(
            self.robot, [0, 0, 0.15-0.001*self.walk._walkPointStartLeftstepRightLeg[:, 0][2]], orientation)

        #startPos = [0.1, -random.uniform(3.0, 3.0), 0.15]
        #p.resetBasePositionAndOrientation(self.robot, startPos, orientation)

        p.setGravity(0, 0, -1)
        self.restSimulation(0.2)
        p.setGravity(0, 0, self.gravity)
        '''
        self.inverseKinematicsMove(
            self.walk._walkPointLeftStepRightLeg[:, self.walk._stepPoint-3], self.walk._walkPointLeftStepLeftLeg[:, self.walk._stepPoint-3], 0, self.walk._stepTime, 0)
        self.inverseKinematicsMove(
            self.walk._walkPointLeftStepRightLeg[:, self.walk._stepPoint-2], self.walk._walkPointLeftStepLeftLeg[:, self.walk._stepPoint-2], 0, self.walk._stepTime, 0)
        self.inverseKinematicsMove(
            self.walk._walkPointLeftStepRightLeg[:, self.walk._stepPoint-1], self.walk._walkPointLeftStepLeftLeg[:, self.walk._stepPoint-1], 0, self.walk._stepTime, 0)
        '''

        for i in range(self.walk._stepPoint):
            self.inverseKinematicsMove(
                self.walk._walkPointStartLeftstepRightLeg[:, i], self.walk._walkPointStartLeftstepLeftLeg[:, i], 0, self.walk._stepTime, 0)

        self.walkRightStep = True

        self.observation = self.compute_observation()
        return np.array(self.observation)


    def change_to_gui_mode(self):
        p.disconnect()
        p.connect(p.GUI)

    def change_to_direct_mode(self):
        p.disconnect()
        p.connect(p.DIRECT)

    def env_setting(self, worker_count, settings):
        self.worker_count = worker_count
        for i in range(self.worker_count):
            self.throwBallEnableList[i] = settings[i][0]
            self.loadTerrain_flat[i] = settings[i][1][0]
            self.loadTerrain_weak_noise[i] = settings[i][1][1]
            self.loadTerrain_strong_noise[i] = settings[i][1][2]
            self.loadTerrain_square[i] = settings[i][1][3] 

            if self.loadTerrain_flat[i] == True:
                self.terrainSwitch[i] = 0
            elif self.loadTerrain_weak_noise[i] == True:
                self.terrainSwitch[i] = 1
            elif self.loadTerrain_strong_noise[i] == True:
                self.terrainSwitch[i] = 2
            else:
                self.terrainSwitch[i] = 3

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        np.clip(action, -1, 1)
        #action = action[0]
        #self.action = action
        # action[0] : 발 높이 차이 더하기. 양수면 뜨는발이 더 위로, 딛는발이 더 아래로.
        # action[1] : 뜨는 발 롤 가중치 (R롤)
        # action[2] : 뜨는 발 피치 가중치 (R피치)
        # action[3] : 딛는 발 롤 가중치 (L롤)
        # action[4] : 딛는 발 피치 가중치 (L피치)
        # print('action_')
        # print(action)
        if(self.walkRightStep):
            self.pos_z_plus += action[0]*10
            if abs(self.pos_z_plus) > 30:
                self.pos_z_plus -= action[0]*10

            self.rightFoot_roll_state += action[1]
            self.rightFoot_pitch_state += action[2]
            self.leftFoot_roll_state += action[3]
            self.leftFoot_pitch_state += action[4]
            if abs(self.rightFoot_roll_state) > 3:
                self.rightFoot_roll_state -= action[1]
            if abs(self.rightFoot_pitch_state) > 3:
                self.rightFoot_pitch_state -= action[1]
            if abs(self.leftFoot_roll_state) > 3:
                self.leftFoot_roll_state -= action[1]
            if abs(self.leftFoot_pitch_state) > 3:
                self.leftFoot_pitch_state -= action[1]

            self.inverseKinematicsMove(
                self.walk._walkPointRightStepRightLeg[:, self.walkStepCounter]+[0, 0, self.pos_z_plus],
                self.walk._walkPointRightStepLeftLeg[:, self.walkStepCounter]+[0, 0, -self.pos_z_plus],
                [0, 0, 0, 0, self.rightFoot_pitch_state, self.rightFoot_roll_state,
                    0, 0, 0, 0, self.leftFoot_pitch_state, self.leftFoot_roll_state],
                self.walk._stepTime, 0)  # self.walk._stepTime

            self.floatingFoot_roll_state_observe = self.rightFoot_roll_state
            self.floatingFoot_pitch_state_observe = self.rightFoot_pitch_state
            self.groundFoot_roll_state_observe = self.leftFoot_roll_state
            self.groundFoot_pitch_state_observe = self.leftFoot_pitch_state

            self.floatingFoot_roll_torque = p.getJointState(self.robot, 6)[3]
            self.floatingFoot_pitch_torque = p.getJointState(self.robot, 5)[3]
            self.groundFoot_roll_torque = p.getJointState(self.robot, 22)[3]
            self.groundFoot_pitch_torque = p.getJointState(self.robot, 21)[3]

            self.walkStepCounter += 1
        else:  # leftStep
            self.pos_z_plus -= action[0]*10
            if abs(self.pos_z_plus) > 30:
                self.pos_z_plus += action[0]*10

            self.rightFoot_roll_state -= action[3]
            self.rightFoot_pitch_state += action[4]
            self.leftFoot_roll_state -= action[1]
            self.leftFoot_pitch_state += action[2]
            if abs(self.rightFoot_roll_state) > 3:
                self.rightFoot_roll_state += action[1]
            if abs(self.rightFoot_pitch_state) > 3:
                self.rightFoot_pitch_state -= action[1]
            if abs(self.leftFoot_roll_state) > 3:
                self.leftFoot_roll_state += action[1]
            if abs(self.leftFoot_pitch_state) > 3:
                self.leftFoot_pitch_state -= action[1]

            self.inverseKinematicsMove(
                self.walk._walkPointLeftStepRightLeg[:, self.walkStepCounter]+[0, 0, self.pos_z_plus],
                self.walk._walkPointLeftStepLeftLeg[:, self.walkStepCounter]+[0, 0, -self.pos_z_plus],
                [0, 0, 0, 0, self.rightFoot_pitch_state, self.rightFoot_roll_state,
                    0, 0, 0, 0, self.leftFoot_pitch_state, self.leftFoot_roll_state],
                self.walk._stepTime, 0)

            self.floatingFoot_roll_state_observe = self.leftFoot_roll_state
            self.floatingFoot_pitch_state_observe = self.leftFoot_pitch_state
            self.groundFoot_roll_state_observe = self.rightFoot_roll_state
            self.groundFoot_pitch_state_observe = self.rightFoot_pitch_state

            self.x_dif = -self.x_dif

            self.floatingFoot_roll_torque = p.getJointState(self.robot, 22)[3]
            self.floatingFoot_pitch_torque = p.getJointState(self.robot, 21)[3]
            self.groundFoot_roll_torque = p.getJointState(self.robot, 6)[3]
            self.groundFoot_pitch_torque = p.getJointState(self.robot, 5)[3]

            self.walkStepCounter += 1
        if(self.walkStepCounter >= self.walk._stepPoint):
            self.walkStepCounter = 0
            if self.walkRightStep == False:
                self.walkRightStep = True
            else:
                self.walkRightStep = False

        if self.throwBallEnable == True:
            if (self.envStepCounter % (self.walk._stepPoint-1)) == 0:
                self.throwBall()
        self.envStepCounter += 1
        self.observation = self.compute_observation()
        # print(self._observation)
        reward = self.compute_reward()
        # print('reward : ', reward)
        done = self.compute_done()
        # print(done)

        # if self.fall_down == True:
        #     reward -= 1
        return self.observation, reward, done, {}

    def compute_observation(self):
        _, cubeOrn = p.getBasePositionAndOrientation(self.robot)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        #jointStateRR_torque = p.getJointState(self.robot, 6)[3]
        #jointStateRP_torque = p.getJointState(self.robot, 5)[3]
        #jointStateLR_torque = p.getJointState(self.robot, 22)[3]
        #jointStateLP_torque = p.getJointState(self.robot, 21)[3]
        #print([cubeEuler[0], cubeEuler[1], jointStateRR, jointStateRP, jointStateLR, jointStateLP])
        # return [cubeEuler[0], cubeEuler[1], jointStateRR, jointStateRP, jointStateLR, jointStateLP]
        '''
        jointStateRR_targetAngle = self.motors._joint_targetPos[5]
        jointStateRP_targetAngle = self.motors._joint_targetPos[4]
        jointStateLR_targetAngle = self.motors._joint_targetPos[11]
        jointStateLP_targetAngle = self.motors._joint_targetPos[10]
        '''
        # jointStateRR = self.action[1]
        # jointStateRP = self.action[2]
        # jointStateLR = self.action[3]
        # jointStateLP = self.action[4]

        # return [cubeEuler[0], cubeEuler[1], self.x_dif, self.pos_z_plus, jointStateRR, jointStateRP, jointStateLR, jointStateLP]
        # return np.array([cubeEuler[0], cubeEuler[1], self.x_dif, self.pos_z_plus, jointStateRR_targetAngle, jointStateRP_targetAngle, jointStateLR_targetAngle, jointStateLP_targetAngle])

        return np.array([cubeEuler[0], cubeEuler[1], self.x_dif, self.pos_z_plus, self.floatingFoot_roll_state_observe, self.floatingFoot_pitch_state_observe, self.groundFoot_roll_state_observe, self.groundFoot_pitch_state_observe, self.floatingFoot_roll_torque, self.floatingFoot_pitch_torque, self.groundFoot_roll_torque, self.groundFoot_pitch_torque])

    def compute_reward(self):
        _, cubeOrn = p.getBasePositionAndOrientation(self.robot)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        # print(math.sqrt(cubeEuler[0]*cubeEuler[0]+cubeEuler[1]*cubeEuler[1]))
        jointStateRR_torque = p.getJointState(self.robot, 6)[3]
        jointStateRP_torque = p.getJointState(self.robot, 5)[3]
        jointStateLR_torque = p.getJointState(self.robot, 22)[3]
        jointStateLP_torque = p.getJointState(self.robot, 21)[3]

        reward = -((cubeEuler[0]*cubeEuler[0]+cubeEuler[1]*cubeEuler[1])*1.0-abs(self.pos_z_plus) * 0.01 - (
            abs(jointStateRR_torque)+abs(jointStateRP_torque) + abs(jointStateLR_torque)+abs(jointStateLP_torque))*0.1)

        #reward = (1-math.sqrt(cubeEuler[0]*cubeEuler[0]+cubeEuler[1]*cubeEuler[1]))*0.0-(self.pos_z_plus * 0.002)
        #reward = 1-(jointStateRR_torque+jointStateRP_torque+jointStateLR_torque+jointStateLP_torque)*0.01
        '''
        self.testCount += 1
        if(self.testCount % 13 == 0):
            print((1-math.sqrt(cubeEuler[0]*cubeEuler[0]+cubeEuler[1]*cubeEuler[1]))*0.1, '\t',
                  (abs(jointStateRR_torque)+abs(jointStateRP_torque)+abs(jointStateLR_torque)+abs(jointStateLP_torque))*0.01, '\t', abs(self.pos_z_plus) * 0.001)
        '''
        return reward

    def compute_done(self):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.robot)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        if (cubeEuler[0]*cubeEuler[0]+cubeEuler[1]*cubeEuler[1]) > self.fall_criteria:
            self.fall_down = True
            #return True
        else:
            self.fall_down = False
        if(self.fall_down):
            print('fall_down')
        self.outPosition = False
        if ((cubePos[0] < -1 or cubePos[0] > 12.4) or (cubePos[1] < self.terrainYpos-12 or cubePos[1] > self.terrainYpos - 0.5)):
            self.outPosition = True
        # return self.fall_down or self.envStepCounter > 2500
        return self.fall_down or (self.envStepCounter > 2500 or self.outPosition)

    def testLoad(self):
        p.resetBasePositionAndOrientation(self.robot, [0, 0, 0.1], p.getQuaternionFromEuler([0, 0, 0]))
    '''
    def reset(self):
        self.envstepCounter = 0
        self.walkStepCounter = 0
        self.loadTerrain()
        p.setGravity(0, 0, -1)
        startPos = [0.1, -1, 0.1]
        orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.robot, [0, 0, 10], orientation)
        self.inverseKinematicsMove(
            self.walk._walkPointLeftStepRightLeg[:, self.walk._stepPoint-4], self.walk._walkPointLeftStepLeftLeg[:, self.walk._stepPoint-4], 0, 0.5, 0)
        p.resetBasePositionAndOrientation(self.robot, startPos, orientation)
        self.restSimulation(0.5)
        p.setGravity(0, 0, self.gravity)
        self.inverseKinematicsMove(
            self.walk._walkPointLeftStepRightLeg[:, self.walk._stepPoint-3], self.walk._walkPointLeftStepLeftLeg[:, self.walk._stepPoint-3], 0, self.walk._stepTime, 0)
        self.inverseKinematicsMove(
            self.walk._walkPointLeftStepRightLeg[:, self.walk._stepPoint-2], self.walk._walkPointLeftStepLeftLeg[:, self.walk._stepPoint-2], 0, self.walk._stepTime, 0)
        self.inverseKinematicsMove(
            self.walk._walkPointLeftStepRightLeg[:, self.walk._stepPoint-1], self.walk._walkPointLeftStepLeftLeg[:, self.walk._stepPoint-1], 0, self.walk._stepTime, 0)
        self.walkRightStep = True
    '''

    def reset(self):
        self.rightFoot_roll_state = 0
        self.rightFoot_pitch_state = 0
        self.leftFoot_roll_state = 0
        self.leftFoot_pitch_state = 0

        self.envStepCounter = 0
        self.walkStepCounter = 0
        self.pos_z_plus = 0
        self.fall_down = False
        if self.setLoadTerrain:
            p.resetSimulation()
            p.setTimeStep(self.timeStep)
            p.setPhysicsEngineParameter(numSolverIterations=self.numSolverIterations)
            self.robot = p.loadURDF(self.urdf_robot_path, [0, 0, 0.2], useFixedBase=False)
            self.motors.setRobot(self.robot)
            #self.ball = p.loadURDF(os.path.join(os.path.abspath(os.path.dirname(__file__)), "ball.xml"), [0, 15, 0.01])
            #plane = p.loadURDF("plane.urdf")
            self.loadTerrain()
        '''
        p.resetSimulation()
        p.setTimeStep(self.timeStep)
        p.setPhysicsEngineParameter(numSolverIterations=self.numSolverIterations)
        self.robot = p.loadURDF(self.urdf_path, useFixedBase=False)
        self.motors.setRobot(self.robot)
        if(self.throwBallEnable == True):
            self.ball = p.loadURDF(os.path.join(os.path.abspath(os.path.dirname(__file__)), "ball.xml"), [0, 10, 1])
        p.setGravity(0, 0, 0)
        self.inverseKinematicsMove(
            self.walk._walkPointLeftStepRightLeg[:, self.walk._stepPoint-4], self.walk._walkPointLeftStepLeftLeg[:, self.walk._stepPoint-4], 0, 0.3, 0.2)
        '''
        orientation = p.getQuaternionFromEuler([0, 0, 0])

        p.setGravity(0, 0, 0)
        #p.resetBasePositionAndOrientation(self.robot, [0, 0, 0.2], orientation)
        self.inverseKinematicsMove(
            self.walk._walkPointStartLeftstepRightLeg[:, 0], self.walk._walkPointStartLeftstepLeftLeg[:, 0], 0, 0.2, 0.3)

        p.resetBasePositionAndOrientation(
            self.robot, [0, 0, 0.15-0.001*self.walk._walkPointStartLeftstepRightLeg[:, 0][2]], orientation)

        #startPos = [0.1, -random.uniform(3.0, 3.0), 0.15]
        #p.resetBasePositionAndOrientation(self.robot, startPos, orientation)

        p.setGravity(0, 0, -1)
        self.restSimulation(0.2)
        p.setGravity(0, 0, self.gravity)
        '''
        self.inverseKinematicsMove(
            self.walk._walkPointLeftStepRightLeg[:, self.walk._stepPoint-3], self.walk._walkPointLeftStepLeftLeg[:, self.walk._stepPoint-3], 0, self.walk._stepTime, 0)
        self.inverseKinematicsMove(
            self.walk._walkPointLeftStepRightLeg[:, self.walk._stepPoint-2], self.walk._walkPointLeftStepLeftLeg[:, self.walk._stepPoint-2], 0, self.walk._stepTime, 0)
        self.inverseKinematicsMove(
            self.walk._walkPointLeftStepRightLeg[:, self.walk._stepPoint-1], self.walk._walkPointLeftStepLeftLeg[:, self.walk._stepPoint-1], 0, self.walk._stepTime, 0)
        '''

        for i in range(self.walk._stepPoint):
            self.inverseKinematicsMove(
                self.walk._walkPointStartLeftstepRightLeg[:, i], self.walk._walkPointStartLeftstepLeftLeg[:, i], 0, self.walk._stepTime, 0)

        self.walkRightStep = True

        self.observation = self.compute_observation()
        return np.array(self.observation)

    def render(self, mode='human', close=False):
        pass

    def setMotorValue(self, kp: float, kd: float, torque: float, speed: float, friction: float):
        self.motor_kp = kp
        self.motor_kd = kd
        self.motor_torque = torque
        self.motor_speed = speed
        self.motor_friction = friction

    def setPhysicsValue(self, gravity, timeStep, numSolverIterations):
        self.gravity = gravity
        self.timeStep = timeStep
        self.numSolverIterations = numSolverIterations

    def normalWalking(self, stepNum):
        self.inverseKinematicsMove(
            self.walk._walkPointStartRightstepRightLeg[:, 0], self.walk._walkPointStartRightstepLeftLeg[:, 0], 0, self.walk._stepTime, 0)

        rightStep = True
        for i in range(self.walk._stepPoint):
            self.inverseKinematicsMove(
                self.walk._walkPointStartRightstepRightLeg[:, i], self.walk._walkPointStartRightstepLeftLeg[:, i], 0, self.walk._stepTime, 0)
        rightStep = False
        for _ in range(stepNum):
            if(rightStep):
                for i in range(self.walk._stepPoint):
                    self.inverseKinematicsMove(
                        self.walk._walkPointRightStepRightLeg[:, i], self.walk._walkPointRightStepLeftLeg[:, i], 0, self.walk._stepTime, 0)
                rightStep = False
            else:
                for i in range(self.walk._stepPoint):
                    self.inverseKinematicsMove(
                        self.walk._walkPointLeftStepRightLeg[:, i], self.walk._walkPointLeftStepLeftLeg[:, i], 0, self.walk._stepTime, 0)
                rightStep = True
        if rightStep == True:
            for i in range(self.walk._stepPoint):
                self.inverseKinematicsMove(
                    self.walk._walkPointEndRightstepRightLeg[:, i], self.walk._walkPointEndRightstepLeftLeg[:, i], 0, self.walk._stepTime, 0)
        else:
            for i in range(self.walk._stepPoint):
                self.inverseKinematicsMove(
                    self.walk._walkPointEndLeftstepRightLeg[:, i], self.walk._walkPointEndLeftstepLeftLeg[:, i], 0, self.walk._stepTime, 0)
        count = 0
        '''
        while(1):
            p.stepSimulation()
            count += 1
            if count > 10000:
                self.compute_reward()
                count = 0
        '''

    def inverseKinematicsMove(self, rightP, LeftP, addAngle, time, delay):
        self.motors.setMotorsAngleInFixedTimestep(
            self.walk.inverseKinematicsPoint(rightP, LeftP) + addAngle, time, delay)
        self.x_dif = (rightP[0]-LeftP[0])
        self.z_dif = rightP[2]-LeftP[2]
        if self.walkRightStep == False:
            self.z_dif = -self.z_dif

    def restSimulation(self, restTime):
        for _ in range(int(restTime/self.timeStep)):
            p.stepSimulation()

    def throwBall(self):
        robotPos, robotOrn = p.getBasePositionAndOrientation(self.robot)
        angle = random.uniform(0, 2)*math.pi
        throwHeight = random.uniform(0.1, 0.5)
        position = [robotPos[0] + math.cos(angle) * (self.throwBallDistance + random.uniform(-0.1, 0.1)),
                    robotPos[1] + math.sin(angle) * (self.throwBallDistance + random.uniform(-0.1, 0.1)),
                    robotPos[2] + throwHeight]
        p.resetBasePositionAndOrientation(self.ball,  position, p.getQuaternionFromEuler([0, 0, 0]))
        force = random.uniform(0.5, 1.0)*self.throwBallForce
        p.applyExternalForce(self.ball, -1, [-math.cos(angle)*force,
                                             -math.sin(angle)*force,
                                             (random.uniform((0.15-throwHeight)*self.throwBallDistance, (0.65-throwHeight)*self.throwBallDistance))*force], [0, 0, 0], p.LINK_FRAME)

    def loadTerrain(self):
        # p.removeBody(self.terrain)

        if self.terrainSwitch == 0:
            if self.loadTerrain_weak_noise == True:
                self.terrainSwitch = 1
            elif self.loadTerrain_strong_noise == True:
                self.terrainSwitch = 2
            elif self.loadTerrain_square == True:
                self.terrainSwitch = 3

            if self.throwBallEnableList[0] == True:
                self.throwBallEnable = True
            else:
                self.throwBallEnable = False
            num = 0
            self.terrainYpos = 6
            startPos = [-0.2, 6, 0.1]
            name = 'flat'
        elif self.terrainSwitch == 1:
            if self.loadTerrain_strong_noise == True:
                self.terrainSwitch = 2
            elif self.loadTerrain_square == True:
                self.terrainSwitch = 3
            elif self.loadTerrain_flat == True:
                self.terrainSwitch = 0

            if self.throwBallEnableList[1] == True:
                self.throwBallEnable = True
            else:
                self.throwBallEnable = False
            num = random.randint(0, 9)
            self.terrainYpos = random.uniform(4.0, 8.0)
            startPos = [-0.2, self.terrainYpos, 0.0]
            name = 'weak_noise'
        elif self.terrainSwitch == 2:
            if self.loadTerrain_square == True:
                self.terrainSwitch = 3
            elif self.loadTerrain_flat == True:
                self.terrainSwitch = 0
            elif self.loadTerrain_weak_noise == True:
                self.terrainSwitch = 1

            if self.throwBallEnableList[2] == True:
                self.throwBallEnable = True
            else:
                self.throwBallEnable = False
            num = random.randint(0, 9)
            self.terrainYpos = random.uniform(4.0, 8.0)
            startPos = [-0.2, self.terrainYpos, 0.0]
            name = 'strong_noise'
        elif self.terrainSwitch == 3:
            if self.loadTerrain_flat == True:
                self.terrainSwitch = 0
            elif self.loadTerrain_weak_noise == True:
                self.terrainSwitch = 1
            elif self.loadTerrain_strong_noise == True:
                self.terrainSwitch = 2

            if self.throwBallEnableList[3] == True:
                self.throwBallEnable = True
            else:
                self.throwBallEnable = False
            num = random.randint(0, 2)
            self.terrainYpos = random.uniform(4.0, 8.0)
            startPos = [-0.2, self.terrainYpos, 0.1]
            name = 'square_noise'

        terrainTXT = open(self.path_running+'/resources'+self.envNumStr+'/terrains/terrainText.txt', 'w')
        terrainString = '''<?xml version="1.0" ?>
<robot name="cube.urdf">
  	<link concave="yes" name="baseLink">
		<inertial>
			<origin rpy="0 0 0" xyz="0 0 0"/>
			<mass value="0"/>
			<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
		</inertial>
		<visual>
			<origin rpy="0 0 0" xyz="0 0 0"/>
			<geometry>
				<mesh filename="'''+name+'/' + str(num)+'''.obj" scale="1 1 1"/>
			</geometry>
			<material name="y">
				<color rgba="0.9 0.9 0.2 1"/>
			</material>
		</visual>
		<collision concave="yes">
			<origin rpy="0 0 0" xyz="0 0 0"/>
			<geometry>
				<mesh filename="'''+name+'/' + str(num)+'''.obj" scale="1 1 1"/>
			</geometry>
		</collision>
	</link>
</robot>'''
        terrainTXT.write(terrainString)
        terrainTXT.close()

        self.terrain = p.loadURDF(self.path_running+'/resources'+self.envNumStr+'/terrains/terrainText.txt',
                                  startPos, p.getQuaternionFromEuler([1.5707963, 0, 1.5707963]), globalScaling=0.025)
        p.changeDynamics(self.terrain, -1,
                         lateralFriction=1.0,
                         spinningFriction=0.8,
                         )


class MotorController:
    def __init__(self, robot, kp, kd, torque, max_velocity, timeStep):
        self._robot = robot
        jointNameToId = {}
        joint_id_list = []
        joint_pos_list = []
        self._joint_number = 0
        for i in range(p.getNumJoints(robot)):
            jointInfo = p.getJointInfo(robot, i)
            if jointInfo[2] == 0:
                joint_id_list.append(jointInfo[0])
                joint_pos_list.append(p.getJointState(robot, jointInfo[0])[0])
                jointNameToId[jointInfo[1].decode('UTF-8')] = jointInfo[0]
                self._joint_number += 1
        self._joint_id = np.array(joint_id_list, dtype=np.int32)
        self._joint_targetPos = np.array(joint_pos_list, dtype=np.float)
        self._joint_currentPos = np.array(joint_pos_list, dtype=np.float)

        self._jointNameToId = jointNameToId
        self._kp = kp
        self._kd = kd
        self._torque = torque
        self._max_velocity = max_velocity
        self._timeStep = timeStep
        # print(self._joint_id)
        # print(self._joint_targetPos)
        print(self._jointNameToId)

    def setRobot(self, robot):
        self._robot = robot
        '''
        joint_id_list = []
        joint_pos_list = []
        self._joint_number = 0
        for i in range(p.getNumJoints(robot)):
            jointInfo = p.getJointInfo(robot, i)
            if jointInfo[2] == 0:
                joint_id_list.append(jointInfo[0])
                joint_pos_list.append(p.getJointState(robot, jointInfo[0])[0])
                self._joint_number += 1
        print(self._joint_number)
        self._joint_id = np.array(joint_id_list, dtype=np.int32)
        self._joint_currentPos = np.array(joint_pos_list, dtype=np.float)
        '''

    def printMotorsAngle(self):
        for i in range(self._joint_number):
            print(list(self._jointNameToId)[i])
            print(p.getJointState(self._robot, self._joint_id[i])[0])
            # self._currentPos[i] = p.getJointState(self._robot, self._joint_id[i])[0]

    def getMotorAngel(self):
        for i in range(self._joint_number):
            self._joint_currentPos[i] = p.getJointState(self._robot, self._joint_id[i])[0]

    def printMotorAngel(self):
        for i in range(self._joint_number):
            self._joint_currentPos[i] = p.getJointState(self._robot, self._joint_id[i])[0]
        print(self._joint_currentPos)

    def setMotorAngle(self, motorTargetAngles):
        for i in range(self._joint_number):
            self._joint_targetPos[i] = motorTargetAngles[i]
            p.setJointMotorControl2(bodyIndex=self._robot, jointIndex=self._joint_id[i], controlMode=p.POSITION_CONTROL,
                                    targetPosition=self._joint_targetPos[i],
                                    positionGain=self._kp, velocityGain=self._kd, force=self._torque, maxVelocity=self._max_velocity)

    def setMotorsAngleInTimestep(self, motorTargetAngles, motorTargetTime, dt):
        for i in range(self._joint_number):
            self._joint_currentPos[i] = p.getJointState(self._robot, self._joint_id[i])[0]
        posPlus = dt*(np.array(motorTargetAngles)-self._joint_currentPos)/motorTargetTime
        moveNum = motorTargetTime/dt

        for _ in range(moveNum):
            for i in range(self._joint_number):
                self._joint_currentPos[i] += posPlus[i]
                p.setJointMotorControl2(bodyIndex=self._robot, jointIndex=self._joint_id[i], controlMode=p.POSITION_CONTROL,
                                        targetPosition=self._joint_currentPos[i],
                                        positionGain=self._kp, velocityGain=self._kd, force=self._torque, maxVelocity=self._max_velocity)

    def setMotorsAngleInRealTime_with_queue(self, q):
        # q : motorTargetAngles, motorTargetTime
        while(True):
            if not q.empty():
                pastTime = 0.0
                motorTargetAngles, motorTargetTime, motorDelayTime = q.get()
                if motorTargetTime == 0:
                    for i in range(self._joint_number):
                        p.setJointMotorControl2(bodyIndex=self._robot, jointIndex=self._joint_id[i], controlMode=p.POSITION_CONTROL,
                                                targetPosition=motorTargetAngles[i],
                                                positionGain=self._kp, velocityGain=self._kd, force=self._torque, maxVelocity=self._max_velocity)
                        time.sleep(motorDelayTime)
                else:
                    refTime = time.time()
                    for i in range(self._joint_number):
                        self._joint_currentPos[i] = p.getJointState(self._robot, self._joint_id[i])[0]
                        dydt = (np.array(motorTargetAngles)-self._joint_currentPos)/motorTargetTime
                    while pastTime < motorTargetTime:
                        pastTime = time.time() - refTime
                        for i in range(self._joint_number):
                            p.setJointMotorControl2(bodyIndex=self._robot, jointIndex=self._joint_id[i], controlMode=p.POSITION_CONTROL,
                                                    targetPosition=self._joint_currentPos[i] + dydt[i] * pastTime,
                                                    positionGain=self._kp, velocityGain=self._kd, force=self._torque, maxVelocity=self._max_velocity)
                    '''
                    for i in range(self._joint_number):
                        p.setJointMotorControl2(bodyIndex=self._robot, jointIndex=self._joint_id[i], controlMode=p.POSITION_CONTROL,
                                                targetPosition=motorTargetAngles[i],
                                                positionGain=self._kp, velocityGain=self._kd, force=self._torque, maxVelocity=self._max_velocity)
                    '''
                    time.sleep(motorDelayTime)

    def setMotorsAngleInRealTimestep(self, motorTargetAngles, motorTargetTime, delayTime):
        if(motorTargetTime == 0):
            self._joint_targetPos = np.array(motorTargetAngles)
            for i in range(self._joint_number):
                p.setJointMotorControl2(bodyIndex=self._robot, jointIndex=self._joint_id[i], controlMode=p.POSITION_CONTROL,
                                        targetPosition=self._joint_targetPos[i],
                                        positionGain=self._kp, velocityGain=self._kd, force=self._torque, maxVelocity=self._max_velocity)
            time.sleep(delayTime)
        else:
            self._joint_currentPos = self._joint_targetPos
            self._joint_targetPos = np.array(motorTargetAngles)
            for i in range(self._joint_number):
                dydt = (self._joint_targetPos-self._joint_currentPos)/motorTargetTime
            internalTime = 0.0
            reft = time.time()
            while internalTime < motorTargetTime:
                internalTime = time.time() - reft
                for i in range(self._joint_number):
                    p.setJointMotorControl2(bodyIndex=self._robot, jointIndex=self._joint_id[i], controlMode=p.POSITION_CONTROL,
                                            targetPosition=self._joint_currentPos[i] + dydt[i] * internalTime,
                                            positionGain=self._kp, velocityGain=self._kd, force=self._torque, maxVelocity=self._max_velocity)

    def setMotorsAngleInFixedTimestep(self, motorTargetAngles, motorTargetTime, delayTime):
        if(motorTargetTime == 0):
            self._joint_targetPos = np.array(motorTargetAngles)
            for i in range(self._joint_number):
                p.setJointMotorControl2(bodyIndex=self._robot, jointIndex=self._joint_id[i], controlMode=p.POSITION_CONTROL,
                                        targetPosition=self._joint_targetPos[i],
                                        positionGain=self._kp, velocityGain=self._kd, force=self._torque, maxVelocity=self._max_velocity)
                p.stepSimulation()
                time.sleep(self._timeStep)
        else:
            self._joint_currentPos = self._joint_targetPos
            self._joint_targetPos = np.array(motorTargetAngles)
            for i in range(self._joint_number):
                dydt = (self._joint_targetPos-self._joint_currentPos)/motorTargetTime
            internalTime = 0.0
            while internalTime < motorTargetTime:
                internalTime += self._timeStep
                i = 0
                while i < (self._joint_number):
                    try:
                        p.setJointMotorControl2(bodyIndex=self._robot, jointIndex=self._joint_id[i], controlMode=p.POSITION_CONTROL,
                                                targetPosition=self._joint_currentPos[i] + dydt[i] * internalTime,
                                                positionGain=self._kp, velocityGain=self._kd, force=self._torque, maxVelocity=self._max_velocity)
                        i += 1
                    except:
                        '''
                        to_print = 'error.\n i                  ' + \
                            str(i) + '\n self._joint_id     ' + str(self._joint_id) + \
                            '\n motorTargetAngles  ' + \
                            str(motorTargetAngles)+'\n self._joint_id[i]   ' + str(self._joint_id[i])
                        print(to_print)
                        '''
                        print('error')
                p.stepSimulation()
                # time.sleep(self._timeStep)
            '''
            for i in range(self._joint_number):
                p.setJointMotorControl2(bodyIndex=self._robot, jointIndex=self._joint_id[i], controlMode=p.POSITION_CONTROL,
                                        targetPosition=motorTargetAngles[i],
                                        positionGain=self._kp, velocityGain=self._kd, force=self._torque, maxVelocity=self._max_velocity)
            p.stepSimulation()
            # time.sleep(timestep)
            '''
            if delayTime != 0:
                for _ in range(int(delayTime/self._timeStep)):
                    p.stepSimulation()


import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


class WalkGenerator():
    def __init__(self):
        #                      R 0   1   2   3   4   5  L6   7   8   9  10  11
        self._motorDirection = [+1, +1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1]
        self._motorDirectionRight = np.array([+1, +1, +1, +1, +1, +1])
        self._motorDirectionLeft = np.array([+1, +1, +1, +1, +1, +1])
        '''
        self._walkPoint0 = 0
        self._walkPoint1 = 0
        self._walkPoint2 = 0
        self._walkPoint3 = 0
        '''
        '''
        self._walkPointStartRightstepRightLeg = 0  # 오른쪽 발을 먼저 내밈. 그때의 오른쪽 발.
        self._walkPointStartRightstepLeftLeg = 0  # 오른쪽 발을 먼저 내밈. 그때의 왼쪽 발.

        self._walkPointStartLeftstepRightLeg = 0  # 왼쪽 발을 먼저 내밈. 그때의 오른쪽 발.
        self._walkPointStartLeftstepLeftLeg = 0  # 왼쪽 발을 먼저 내밈. 그때의 왼쪽 발.

        self._walkPointEndRightstepRightLeg = 0  # 오른쪽 발을 디디면서 끝남. 그때의 오른쪽 발.
        self._walkPointEndLeftstepRightLeg = 0  # 오른쪽 발을 디디면서 끝남. 그때의 왼쪽 발.
        self._walkPointEndLeftstepLeftLeg = 0
        self._walkPointEndRightstepLeftLeg = 0
        # self._walkPoint1fInverse = 0
        # self._walkPoint2fInverse = 0

        self._walkPointRightStepRightLeg = 0
        self._walkPointLeftStepRightLeg = 0

        self._walkPointRightStepLeftLeg = 0
        self._walkPointLeftStepLeftLeg = 0

        self._walkPointRightStepInverse = 0
        self._walkPointLeftStepInverse = 0

        self._walkPointStartRightInverse = 0  # 왼쪽으로 sway 했다가 오른발을 먼저 내밈.
        self._walkPointStartLeftInverse = 0  # 오른쪽으로 sway 했다가 왼발을 먼저 내밈.
        self._walkPointEndRightInverse = 0  # 오른발을 디디면서 끝남.
        self._walkPointEndLeftInverse = 0  # 왼발을 디디면서 끝남.

        self._walkPoint0AnkleX = 0
        self._walkPoint1AnkleX = 0
        self._walkPoint2AnkleX = 0
        self._walkPoint3AnkleX = 0

        self._turnListUnfold = 0
        self._turnListFold = 0
        '''
        # 로봇의 길이 설정. 길이 단위: mm
        self._pelvic_interval = 70.5
        self._legUp_length = 110
        self._legDown_length = 110
        self._footJoint_to_bottom = 45
        '''
        self._bodyMovePoint = 0
        self._legMovePoint = 0
        self._h = 0
        self._l = 0
        self._sit = 0
        self._swayBody = 0
        self._swayFoot = 0
        self._swayShift = 0
        self._weightStart = 0
        self._weightEnd = 0
        self._stepTime = 0
        self._bodyPositionXPlus = 0
        self._damping = 0
        '''

    def setRobotParameter(self, pelvic_interval, leg_up_length, leg_down_length, foot_to_grount, foot_to_heel, foot_to_toe):
        pass

    def setWalkParameter(self, bodyMovePoint, legMovePoint, h, l, sit, swayBody, swayFoot, bodyPositionXPlus, swayShift, weightStart, weightEnd, stepTime, damping, incline):
        self._bodyMovePoint = bodyMovePoint
        self._legMovePoint = legMovePoint
        self._h = h
        self._l = l
        self._sit = sit
        self._swayBody = swayBody
        self._swayFoot = swayFoot
        self._swayShift = swayShift
        self._weightStart = weightStart
        self._weightEnd = weightEnd
        self._stepTime = stepTime
        self._bodyPositionXPlus = bodyPositionXPlus  # +면 몸을 앞으로, -면 몸을 뒤로 하고 걸음.
        self._damping = damping
        self._incline = incline

        self._stepPoint = bodyMovePoint+legMovePoint

    def generate(self):
        walkPoint = self._bodyMovePoint*2+self._legMovePoint*2
        trajectoryLength = self._l*(2*self._bodyMovePoint + self._legMovePoint) / \
            (self._bodyMovePoint + self._legMovePoint)
        print('trajectoryLength')
        print(trajectoryLength)

        walkPoint0 = np.zeros((3, self._bodyMovePoint))
        walkPoint1 = np.zeros((3, self._legMovePoint))
        walkPoint2 = np.zeros((3, self._bodyMovePoint))
        walkPoint3 = np.zeros((3, self._legMovePoint))

        self._walkPointStartRightstepRightLeg = np.zeros((3, self._bodyMovePoint+self._legMovePoint))
        self._walkPointStartLeftstepRightLeg = np.zeros((3, self._bodyMovePoint+self._legMovePoint))
        self._walkPointEndRightstepRightLeg = np.zeros((3, self._bodyMovePoint+self._legMovePoint))
        self._walkPointEndLeftstepRightLeg = np.zeros((3, self._bodyMovePoint+self._legMovePoint))

        for i in range(self._bodyMovePoint):
            t = (i+1)/(walkPoint-self._legMovePoint)
            walkPoint0[0][i] = -trajectoryLength*(t-0.5)
            walkPoint0[2][i] = self._sit
            walkPoint0[1][i] = self._swayBody*math.sin(2 * math.pi*((i+1-self._swayShift)/walkPoint))

        for i in range(self._legMovePoint):
            t = (i+1 + self._bodyMovePoint)/(walkPoint-self._legMovePoint)
            walkPoint1[0][i] = -trajectoryLength*(t-0.5)
            walkPoint1[2][i] = self._sit
            walkPoint1[1][i] = self._swayBody * \
                math.sin(2 * math.pi*((i + 1 + self._bodyMovePoint-self._swayShift)/walkPoint))

        for i in range(self._bodyMovePoint):
            t = (i + 1 + self._bodyMovePoint+self._legMovePoint)/(walkPoint-self._legMovePoint)
            walkPoint2[0][i] = -trajectoryLength*(t-0.5)
            walkPoint2[2][i] = self._sit
            walkPoint2[1][i] = self._swayBody * \
                math.sin(2 * math.pi*((i + 1 + self._bodyMovePoint+self._legMovePoint-self._swayShift)/walkPoint))

        for i in range(self._legMovePoint):
            t = (i+1) / self._legMovePoint
            sin_tpi = math.sin(t * math.pi)

            walkPoint3[0][i] = (2 * t - 1 + (1-t) * self._weightStart * -sin_tpi +
                                t * self._weightEnd * sin_tpi) * trajectoryLength / 2
            walkPoint3[2][i] = math.sin(t * math.pi) * self._h + self._sit
            walkPoint3[1][i] = math.sin(t * math.pi) * self._swayFoot + self._swayBody * \
                math.sin(2 * math.pi*((i+1+walkPoint-self._legMovePoint-self._swayShift)/walkPoint))

        # 시작 동작 만들기
        for i in range(self._bodyMovePoint-self._swayShift):
            t = (i+1)/self._bodyMovePoint
            self._walkPointStartRightstepRightLeg[0][i] = 0
            self._walkPointStartRightstepRightLeg[2][i] = self._sit

            self._walkPointStartLeftstepRightLeg[0][i] = 0
            self._walkPointStartLeftstepRightLeg[2][i] = self._sit
        for i in range(self._legMovePoint):
            t = (i+1)/self._legMovePoint
            t2 = (i+1)/(self._legMovePoint+self._swayShift)
            sin_tpi = math.sin(t * math.pi)

            self._walkPointStartRightstepRightLeg[2][i+self._bodyMovePoint -
                                                     self._swayShift] = math.sin(t * math.pi) * self._h + self._sit
            self._walkPointStartRightstepRightLeg[0][i+self._bodyMovePoint - self._swayShift] = (
                2 * t + (1-t) * self._weightStart * -sin_tpi + t * self._weightEnd * sin_tpi) * trajectoryLength / 4
            self._walkPointStartLeftstepRightLeg[0][i+self._bodyMovePoint-self._swayShift] = (math.cos(
                t2*math.pi/2)-1) * trajectoryLength * self._legMovePoint/(self._bodyMovePoint*2+self._legMovePoint)/2
            self._walkPointStartLeftstepRightLeg[0][i+self._bodyMovePoint-self._swayShift] = (math.cos(t2*math.pi/2)-1) * trajectoryLength * (
                (self._swayShift+self._bodyMovePoint+self._legMovePoint)/(self._bodyMovePoint*2+self._legMovePoint)-0.5)

            self._walkPointStartLeftstepRightLeg[2][i+self._bodyMovePoint-self._swayShift] = self._sit

        for i in range(self._swayShift):
            t2 = (i+1+self._legMovePoint)/(self._legMovePoint+self._swayShift)

            self._walkPointStartRightstepRightLeg[0][i+self._legMovePoint+self._bodyMovePoint-self._swayShift] = - \
                trajectoryLength*((i+1)/(walkPoint-self._legMovePoint)-0.5)
            self._walkPointStartRightstepRightLeg[2][i+self._legMovePoint +
                                                     self._bodyMovePoint-self._swayShift] = self._sit

            self._walkPointStartLeftstepRightLeg[0][i+self._legMovePoint+self._bodyMovePoint-self._swayShift] = - \
                trajectoryLength*((i + 1 + self._bodyMovePoint+self._legMovePoint)/(walkPoint-self._legMovePoint)-0.5)

            self._walkPointStartLeftstepRightLeg[0][i+self._legMovePoint+self._bodyMovePoint-self._swayShift] = (math.cos(t2*math.pi/2)-1) * trajectoryLength * (
                (self._swayShift+self._bodyMovePoint+self._legMovePoint)/(self._bodyMovePoint*2+self._legMovePoint)-0.5)

            self._walkPointStartLeftstepRightLeg[2][i+self._legMovePoint +
                                                    self._bodyMovePoint-self._swayShift] = self._sit

        for i in range(self._bodyMovePoint+self._legMovePoint):
            t = (i+1)/(self._bodyMovePoint+self._legMovePoint)
            #self._walkPointStartRightstepRightLeg[1][i] = -self._swayBody * math.sin(t*math.pi) * math.sin(t*math.pi)
            #self._walkPointStartLeftstepRightLeg[1][i] = self._swayBody * math.sin(t*math.pi) * math.sin(t*math.pi)
            if t < 1/4:
                self._walkPointStartRightstepRightLeg[1][i] = -self._swayBody * \
                    (math.sin(t*math.pi) - (1-math.sin(math.pi*2*t))*(math.sin(4*t*math.pi)/4))
                self._walkPointStartLeftstepRightLeg[1][i] = self._swayBody * \
                    (math.sin(t*math.pi) - (1-math.sin(math.pi*2*t))*(math.sin(4*t*math.pi)/4))
            else:
                self._walkPointStartRightstepRightLeg[1][i] = -self._swayBody * math.sin(t*math.pi)
                self._walkPointStartLeftstepRightLeg[1][i] = self._swayBody * math.sin(t*math.pi)

        # 마무리 동작 만들기. 왼발이 뜸. 그러나 둘다 오른쪽다리 기준
        for i in range(self._bodyMovePoint-self._swayShift):
            self._walkPointEndLeftstepRightLeg[0][i] = -trajectoryLength * \
                ((i+1+self._swayShift)/(walkPoint-self._legMovePoint)-0.5)
            self._walkPointEndLeftstepRightLeg[2][i] = self._sit

            self._walkPointEndRightstepRightLeg[0][i] = -trajectoryLength * \
                ((i + 1 + self._swayShift + self._bodyMovePoint+self._legMovePoint)/(walkPoint-self._legMovePoint)-0.5)
            self._walkPointEndRightstepRightLeg[2][i] = self._sit
        for i in range(self._legMovePoint):
            t = (i+1)/self._legMovePoint
            sin_tpi = math.sin(t * math.pi)

            self._walkPointEndLeftstepRightLeg[0][i+self._bodyMovePoint-self._swayShift] = (math.sin(t*math.pi/2)-1) * trajectoryLength * (
                (self._bodyMovePoint)/(self._bodyMovePoint*2+self._legMovePoint)-0.5)
            self._walkPointEndLeftstepRightLeg[2][i+self._bodyMovePoint-self._swayShift] = self._sit

            self._walkPointEndRightstepRightLeg[0][i+self._bodyMovePoint-self._swayShift] = (
                2 * t-2 + (1-t) * self._weightStart * -sin_tpi + t * self._weightEnd * sin_tpi) * trajectoryLength / 4
            self._walkPointEndRightstepRightLeg[2][i+self._bodyMovePoint -
                                                   self._swayShift] = math.sin(t * math.pi) * self._h + self._sit
        for i in range(self._swayShift):
            self._walkPointEndLeftstepRightLeg[0][i+self._bodyMovePoint+self._legMovePoint-self._swayShift] = 0
            self._walkPointEndLeftstepRightLeg[2][i+self._bodyMovePoint+self._legMovePoint-self._swayShift] = self._sit

            self._walkPointEndRightstepRightLeg[0][i+self._bodyMovePoint+self._legMovePoint-self._swayShift] = 0
            self._walkPointEndRightstepRightLeg[2][i+self._bodyMovePoint+self._legMovePoint-self._swayShift] = self._sit

        # turnList

        self._turnListUnfold = np.zeros((self._bodyMovePoint+self._legMovePoint, 12))
        self._turnListFold = np.zeros((self._bodyMovePoint+self._legMovePoint, 12))
        turnAngle = np.zeros(self._bodyMovePoint+self._legMovePoint)
        for i in range(self._legMovePoint):
            t = (i+1)/self._legMovePoint
            turnAngle[self._bodyMovePoint-self._swayShift + i] = (1-math.cos(math.pi*t))/4
        for i in range(self._swayShift):
            turnAngle[self._bodyMovePoint+self._legMovePoint-self._swayShift + i] = 1/2

        for i in range(self._bodyMovePoint+self._legMovePoint):
            self._turnListUnfold[i] = [turnAngle[i], 0, 0, 0, 0, 0, -turnAngle[i], 0, 0, 0, 0, 0]
            self._turnListFold[i] = [0.5-turnAngle[i], 0, 0, 0, 0, 0, -0.5+turnAngle[i], 0, 0, 0, 0, 0]

        for i in range(self._bodyMovePoint+self._legMovePoint):
            t = 1 - (i+1)/(self._bodyMovePoint+self._legMovePoint)

            if t < 1/4:
                self._walkPointEndLeftstepRightLeg[1][i] = self._swayBody * \
                    (math.sin(t*math.pi) - (1-math.sin(math.pi*2*t))*(math.sin(4*t*math.pi)/4))
                self._walkPointEndRightstepRightLeg[1][i] = -self._swayBody * \
                    (math.sin(t*math.pi) - (1-math.sin(math.pi*2*t))*(math.sin(4*t*math.pi)/4))
            else:
                self._walkPointEndLeftstepRightLeg[1][i] = self._swayBody * math.sin(t*math.pi)
                self._walkPointEndRightstepRightLeg[1][i] = -self._swayBody * math.sin(t*math.pi)

        # 추가 파라미터의 조정

        if self._incline != 0:  # 기울기. 계단 등에서 사용.
            walkPoint0[2] = walkPoint0[2] + walkPoint0[0]*self._incline
            walkPoint1[2] = walkPoint1[2] + walkPoint1[0]*self._incline
            walkPoint2[2] = walkPoint2[2] + walkPoint2[0]*self._incline
            walkPoint3[2] = walkPoint3[2] + walkPoint3[0]*self._incline
            self._walkPointStartRightstepRightLeg[2] = self._walkPointStartRightstepRightLeg[2] + \
                self._walkPointStartRightstepRightLeg[0]*self._incline
            self._walkPointStartLeftstepRightLeg[2] = self._walkPointStartLeftstepRightLeg[2] + \
                self._walkPointStartLeftstepRightLeg[0]*self._incline
            self._walkPointEndLeftstepRightLeg[2] = self._walkPointEndLeftstepRightLeg[2] + \
                self._walkPointEndLeftstepRightLeg[0]*self._incline
            self._walkPointEndRightstepRightLeg[2] = self._walkPointEndRightstepRightLeg[2] + \
                self._walkPointEndRightstepRightLeg[0]*self._incline

        if self._bodyPositionXPlus != 0:  # 허리 앞뒤 위치 조절
            walkPoint0[0] = walkPoint0[0] - self._bodyPositionXPlus
            walkPoint1[0] = walkPoint1[0] - self._bodyPositionXPlus
            walkPoint2[0] = walkPoint2[0] - self._bodyPositionXPlus
            walkPoint3[0] = walkPoint3[0] - self._bodyPositionXPlus
            self._walkPointStartRightstepRightLeg[0] = self._walkPointStartRightstepRightLeg[0] - \
                self._bodyPositionXPlus
            self._walkPointStartLeftstepRightLeg[0] = self._walkPointStartLeftstepRightLeg[0] - self._bodyPositionXPlus
            self._walkPointEndLeftstepRightLeg[0] = self._walkPointEndLeftstepRightLeg[0] - self._bodyPositionXPlus
            self._walkPointEndRightstepRightLeg[0] = self._walkPointEndRightstepRightLeg[0] - self._bodyPositionXPlus

        if self._damping != 0:  # 댐핑 조절
            dampHeight = (walkPoint3[2][-1]-walkPoint0[2][0])/2
            walkPoint0[2][0] = walkPoint0[2][0]+dampHeight*self._damping
            walkPoint2[2][0] = walkPoint2[2][0]-dampHeight*self._damping

        self._walkPoint0 = walkPoint0
        self._walkPoint1 = walkPoint1
        self._walkPoint2 = walkPoint2
        self._walkPoint3 = walkPoint3

        self._walkPointLeftStepRightLeg = np.column_stack(
            [walkPoint0[:, self._swayShift:], walkPoint1, walkPoint2[:, :self._swayShift]])
        self._walkPointRightStepRightLeg = np.column_stack(
            [walkPoint2[:, self._swayShift:], walkPoint3, walkPoint0[:, :self._swayShift]])

        self._walkPointLeftStepLeftLeg = self._walkPointRightStepRightLeg * np.array([[1], [-1], [1]])
        self._walkPointRightStepLeftLeg = self._walkPointLeftStepRightLeg*np.array([[1], [-1], [1]])

        self._walkPointStartRightstepLeftLeg = self._walkPointStartLeftstepRightLeg * np.array([[1], [-1], [1]])
        self._walkPointStartLeftstepLeftLeg = self._walkPointStartRightstepRightLeg * np.array([[1], [-1], [1]])

        self._walkPointEndLeftstepLeftLeg = self._walkPointEndRightstepRightLeg * np.array([[1], [-1], [1]])
        self._walkPointEndRightstepLeftLeg = self._walkPointEndLeftstepRightLeg * np.array([[1], [-1], [1]])

    def inverseKinematicsList(self, point, isRightLeg):
        inverseAngle = np.zeros((point[0].size, 6))
        for i in range(point[0].size):
            l3 = self._legUp_length
            l4 = self._legDown_length

            fx = point[0][i]
            fy = point[1][i]
            fz = self._legUp_length + self._legDown_length - point[2][i]

            a = math.sqrt(fx*fx + fy * fy + fz * fz)

            d1 = math.asin(fx/a)
            d2 = math.acos((l3*l3+a*a-l4*l4)/(2*l3*a))
            #d3 = math.acos(fz/a)
            d4 = math.acos((l4*l4+a*a-l3*l3)/(2*l4*a))
            d5 = math.pi-d2-d4

            t1 = (math.atan2(fy, fz))
            t2 = d1+d2
            t3 = math.pi-d5
            t4 = -t2+t3
            t5 = -t1

            if isRightLeg:
                inverseAngle[i] = np.array([0, t1, -t2, t3, -t4, t5]) * self._motorDirectionRight
            else:
                inverseAngle[i] = np.array([0, t1, -t2, t3, -t4, t5]) * self._motorDirectionLeft

        return inverseAngle

    def inverseKinematicsPoint(self, pointRight, pointLeft):
        l3 = self._legUp_length
        l4 = self._legDown_length

        fx = pointRight[0]
        fy = pointRight[1]
        fz = self._legUp_length + self._legDown_length - pointRight[2]

        a = math.sqrt(fx*fx + fy * fy + fz * fz)

        d1 = math.asin(fx/a)
        d2 = math.acos((l3*l3+a*a-l4*l4)/(2*l3*a))
        #d3 = math.acos(fz/a)
        d4 = math.acos((l4*l4+a*a-l3*l3)/(2*l4*a))
        d5 = math.pi-d2-d4

        t1 = (math.atan2(fy, fz))
        t2 = d1+d2
        t3 = math.pi-d5
        t4 = -t2+t3
        t5 = -t1

        rightInverse = np.array([0, t1, -t2, t3, -t4, t5]) * self._motorDirectionRight

        fx = pointLeft[0]
        fy = pointLeft[1]
        fz = self._legUp_length + self._legDown_length - pointLeft[2]

        a = math.sqrt(fx*fx + fy * fy + fz * fz)

        d1 = math.asin(fx/a)
        d2 = math.acos((l3*l3+a*a-l4*l4)/(2*l3*a))
        #d3 = math.acos(fz/a)
        d4 = math.acos((l4*l4+a*a-l3*l3)/(2*l4*a))
        d5 = math.pi-d2-d4

        t1 = (math.atan2(fy, fz))
        t2 = d1+d2
        t3 = math.pi-d5
        t4 = -t2+t3
        t5 = -t1

        leftInverse = np.array([0, t1, -t2, t3, -t4, t5]) * self._motorDirectionLeft
        #print(np.hstack([rightInverse, leftInverse]))
        #print(np.hstack([rightInverse, leftInverse]))
        return np.hstack([rightInverse, leftInverse])

    def showGaitPoint2D(self):
        plt.plot(self._walkPoint0[0], self._walkPoint0[2], 'o-', c='red',  ms=7, lw=5)
        plt.plot(self._walkPoint1[0], self._walkPoint1[2], 'o-', c='blue', ms=7, lw=5)
        plt.plot(self._walkPoint2[0], self._walkPoint2[2], 'o-', c='red',  ms=7, lw=5)
        plt.plot(self._walkPoint3[0], self._walkPoint3[2], 'o-', c='blue', ms=7, lw=5)

        plt.plot(self._walkPointStartRightstepRightLeg[0], self._walkPointStartRightstepRightLeg[2], '*-')
        plt.plot(self._walkPointStartLeftstepRightLeg[0], self._walkPointStartLeftstepRightLeg[2],   '*-')
        plt.plot(self._walkPointEndRightstepRightLeg[0], self._walkPointEndRightstepRightLeg[2],     '*-')
        plt.plot(self._walkPointEndLeftstepRightLeg[0], self._walkPointEndLeftstepRightLeg[2],       '*-')

        plt.show()

    def showGaitPoint2DTop(self):
        plt.plot(self._walkPoint0[0], self._walkPoint0[1], 'o-')
        plt.plot(self._walkPoint1[0], self._walkPoint1[1], 'o-')
        plt.plot(self._walkPoint2[0], self._walkPoint2[1], 'o-')
        plt.plot(self._walkPoint3[0], self._walkPoint3[1], 'o-')

        plt.plot(self._walkPointStartRightstepRightLeg[0], self._walkPointStartRightstepRightLeg[1], '.-')
        plt.plot(self._walkPointStartLeftstepRightLeg[0], self._walkPointStartLeftstepRightLeg[1], '.-')

        plt.plot(self._walkPointEndRightstepRightLeg[0], self._walkPointEndRightstepRightLeg[1], '+-')
        plt.plot(self._walkPointEndLeftstepRightLeg[0], self._walkPointEndLeftstepRightLeg[1], '+-')

        plt.show()

    def showGaitPoint3D(self):
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(self._walkPointRightStepRightLeg[0], self._walkPointRightStepRightLeg[1],
                self._walkPointRightStepRightLeg[2], 'co-', lw=10, ms=6)
        ax.plot(self._walkPointLeftStepRightLeg[0], self._walkPointLeftStepRightLeg[1],
                self._walkPointLeftStepRightLeg[2], 'mo-', lw=10, ms=6)

        ax.plot(self._walkPoint0[0], self._walkPoint0[1], self._walkPoint0[2], 'o')
        ax.plot(self._walkPoint1[0], self._walkPoint1[1], self._walkPoint1[2], 'o')
        ax.plot(self._walkPoint2[0], self._walkPoint2[1], self._walkPoint2[2], 'o')
        ax.plot(self._walkPoint3[0], self._walkPoint3[1], self._walkPoint3[2], 'o')

        ax.plot(self._walkPointStartRightstepRightLeg[0], self._walkPointStartRightstepRightLeg[1],
                self._walkPointStartRightstepRightLeg[2], '*-')
        ax.plot(self._walkPointStartLeftstepRightLeg[0], self._walkPointStartLeftstepRightLeg[1],
                self._walkPointStartLeftstepRightLeg[2], '*-')

        ax.plot(self._walkPointEndRightstepRightLeg[0], self._walkPointEndRightstepRightLeg[1],
                self._walkPointEndRightstepRightLeg[2], 'c-')
        ax.plot(self._walkPointEndLeftstepRightLeg[0], self._walkPointEndLeftstepRightLeg[1],
                self._walkPointEndLeftstepRightLeg[2], '+-')

        plt.show()

    def inverseKinematicsAll(self):
        self._walkPointStartRightInverse = np.column_stack(
            [self.inverseKinematicsList(self._walkPointStartRightstepRightLeg, True),
             self.inverseKinematicsList(self._walkPointStartRightstepLeftLeg, False)])
        self._walkPointStartLeftInverse = np.column_stack(
            [self.inverseKinematicsList(self._walkPointStartLeftstepRightLeg, True),
             self.inverseKinematicsList(self._walkPointStartLeftstepLeftLeg, False)])

        self._walkPointEndLeftInverse = np.column_stack(
            [self.inverseKinematicsList(self._walkPointEndLeftstepRightLeg, True),
             self.inverseKinematicsList(self._walkPointEndLeftstepLeftLeg, False)])
        self._walkPointEndRightInverse = np.column_stack(
            [self.inverseKinematicsList(self._walkPointEndRightstepRightLeg, True),
             self.inverseKinematicsList(self._walkPointEndRightstepLeftLeg, False)])
        # self._walkPointStartLeftInverse = walkpointstartLeft_inverse
        self._walkPointRightStepInverse = np.column_stack(
            [self.inverseKinematicsList(self._walkPointRightStepRightLeg, True),
             self.inverseKinematicsList(self._walkPointRightStepLeftLeg, False)])
        self._walkPointLeftStepInverse = np.column_stack(
            [self.inverseKinematicsList(self._walkPointLeftStepRightLeg, True),
             self.inverseKinematicsList(self._walkPointLeftStepLeftLeg, False)])
