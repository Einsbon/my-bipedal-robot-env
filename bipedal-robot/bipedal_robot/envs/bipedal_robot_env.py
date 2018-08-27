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

from . import walkGenerator
from . import motorController

import matplotlib.pyplot as plt
import time


class BipedalRobotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        # 물리 값 설정
        self.gravity = -9.8
        self.timeStep = 1.0/400
        self.numSolverIterations = 200
        self.friction = 0.8
        self.spinFrictino = 0.8

        self.footfriction = 0.8
        self.perturbation = 0.05

        self.path_running = os.path.abspath(os.path.dirname(__file__))

        self.reset_physicsWorld_when_reset = True
        isDirectConnect = True
        # 물리 엔진 셋업
        if isDirectConnect == True:
            self.physicsClientId = p.connect(p.DIRECT)
        else:
            self.physicsClientId = p.connect(p.GUI)
        p.setTimeStep(self.timeStep, physicsClientId=self.physicsClientId)
        p.setPhysicsEngineParameter(numSolverIterations=self.numSolverIterations, physicsClientId=self.physicsClientId)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.physicsClientId)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physicsClientId)
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
        self.fall_criteria = 0.3
        # # 지형 설정
        self.setLoadTerrain = True
        self.loadTerrain_flat = False  # 0
        self.loadTerrain_weak_noise = False  # 1
        self.loadTerrain_strong_noise = True  # 2
        self.loadTerrain_square = False  # 3
        self.terrainSwitch = 0
        # # 공던지기 설정
        self.ballPath = os.path.abspath(os.path.dirname(__file__)) + '/resources'+'/ball.xml'
        self.ball = p.loadURDF(self.ballPath, [0, 20, 0.5], useFixedBase=False, physicsClientId=self.physicsClientId)
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
        self.urdf_robot_path = os.path.abspath(os.path.dirname(__file__)) + '/resources'+'/humanoid_leg_12dof.7.urdf'

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
        self.walk = walkGenerator.WalkGenerator()
        self.walk.setWalkParameter(bodyMovePoint=6, legMovePoint=6, h=50, l=90, sit=50, swayBody=45, swayFoot=0,
                                   bodyPositionXPlus=5, swayShift=3, weightStart=0.5, weightEnd=0.7, stepTime=0.08, damping=0.0, incline=0.0)
        self.walk.generate()
        # self.walk.showGaitPoint3D()
        # self.walk.inverseKinematicsAll()

        self.rightFoot_roll_state = 0
        self.rightFoot_pitch_state = 0
        self.leftFoot_roll_state = 0
        self.leftFoot_pitch_state = 0

        self.floatingFoot_roll_state_observe = 0
        self.floatingFoot_pitch_state_observe = 0
        self.groundFoot_roll_state_observe = 0
        self.groundFoot_pitch_state_observe = 0

        self.floatingFoot_roll_torque = 0
        self.floatingFoot_pitch_torque = 0
        self.groundFoot_roll_torque = 0
        self.groundFoot_pitch_torque = 0

        self.terrainYpos = 0

        self.reward = 0

        self.testCount = 0
        self.envStepCounter = 0

        self.envNum = 0
        self.envNumStr = ''

        self.robot = p.loadURDF(self.urdf_robot_path, [0, 0, 1],
                                useFixedBase=False, physicsClientId=self.physicsClientId)
        self.motors = motorController.MotorController(
            self.robot, self.physicsClientId, self.timeStep, self.motor_kp, self.motor_kd, self.motor_torque, self.motor_speed)
        self.loadTerrain()

        self.seed()

    def setEnvironmentNumber(self, num):
        self.envNum = num
        self.envNumStr = str(num)
        self.urdf_robot_path = os.path.abspath(os.path.dirname(__file__)) + \
            '/resources' + self.envNumStr + '/humanoid_leg_12dof.7.urdf'
        self.ballPath = os.path.abspath(os.path.dirname(__file__)) + '/resources' + self.envNumStr + '/ball.xml'
        self.robot = p.loadURDF(self.urdf_robot_path, [0, 0, 1],
                                useFixedBase=False, physicsClientId=self.physicsClientId)
        self.motors = motorController.MotorController(
            self.robot, self.physicsClientId, self.timeStep, self.motor_kp, self.motor_kd, self.motor_torque, self.motor_speed)

    def change_to_gui_mode(self):
        p.disconnect(self.physicsClientId)
        self.physicsClientId = p.connect(p.GUI)

        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-30, cameraPitch=-5,
                                     cameraTargetPosition=[0.5, 0, 0.3], physicsClientId=self.physicsClientId)

    def change_to_direct_mode(self):
        p.disconnect(self.physicsClientId)
        self.physicsClientId = p.connect(p.DIRECT)

    def env_setting(self, throwBallEnableList, loadTerrain0, loadTerrain1, loadTerrain2, loadTerrain3):
        self.throwBallEnableList = throwBallEnableList
        self.loadTerrain_flat = loadTerrain0
        self.loadTerrain_weak_noise = loadTerrain1
        self.loadTerrain_strong_noise = loadTerrain2
        self.loadTerrain_square = loadTerrain3
        if self.loadTerrain_flat == True:
            self.terrainSwitch = 0
        elif self.loadTerrain_weak_noise == True:
            self.terrainSwitch = 1
        elif self.loadTerrain_strong_noise == True:
            self.terrainSwitch = 2
        else:
            self.terrainSwitch = 3

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, act):
        action = np.clip(act, -1, 1)
        action = action * np.array([10, 0.2, 0.2, 0.2, 0.2])
        # action = action[0]
        # self.action = action
        # action[0] : 발 높이 차이 더하기. 양수면 뜨는발이 더 위로, 딛는발이 더 아래로.
        # action[1] : 뜨는 발 롤 가중치 (R롤)
        # action[2] : 뜨는 발 피치 가중치 (R피치)
        # action[3] : 딛는 발 롤 가중치 (L롤)
        # action[4] : 딛는 발 피치 가중치 (L피치)
        # print('action_')
        # print(action)
        if(self.walkRightStep):
            self.pos_z_plus += action[0]
            if (self.pos_z_plus) < -30:
                self.pos_z_plus = -30
            elif self.pos_z_plus > 30:
                self.pos_z_plus = 30

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
                self.walk.walkPointRightStepRightLeg[:, self.walkStepCounter]+[0, 0, self.pos_z_plus],
                self.walk.walkPointRightStepLeftLeg[:, self.walkStepCounter]+[0, 0, -self.pos_z_plus],
                [0, 0, 0, 0, self.rightFoot_pitch_state, self.rightFoot_roll_state,
                    0, 0, 0, 0, self.leftFoot_pitch_state, self.leftFoot_roll_state],
                self.walk._stepTime, 0)  # self.walk._stepTime

            self.floatingFoot_roll_state_observe = self.rightFoot_roll_state
            self.floatingFoot_pitch_state_observe = self.rightFoot_pitch_state
            self.groundFoot_roll_state_observe = self.leftFoot_roll_state
            self.groundFoot_pitch_state_observe = self.leftFoot_pitch_state

            self.floatingFoot_roll_torque = p.getJointState(self.robot, 6, physicsClientId=self.physicsClientId)[3]
            self.floatingFoot_pitch_torque = p.getJointState(self.robot, 5, physicsClientId=self.physicsClientId)[3]
            self.groundFoot_roll_torque = p.getJointState(self.robot, 22, physicsClientId=self.physicsClientId)[3]
            self.groundFoot_pitch_torque = p.getJointState(self.robot, 21, physicsClientId=self.physicsClientId)[3]

            self.walkStepCounter += 1
        else:  # leftStep
            self.pos_z_plus -= action[0]*10
            if (self.pos_z_plus) < -30:
                self.pos_z_plus = -30
            elif self.pos_z_plus > 30:
                self.pos_z_plus = 30

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
                self.walk.walkPointLeftStepRightLeg[:, self.walkStepCounter]+[0, 0, self.pos_z_plus],
                self.walk.walkPointLeftStepLeftLeg[:, self.walkStepCounter]+[0, 0, -self.pos_z_plus],
                [0, 0, 0, 0, self.rightFoot_pitch_state, self.rightFoot_roll_state,
                    0, 0, 0, 0, self.leftFoot_pitch_state, self.leftFoot_roll_state],
                self.walk._stepTime, 0)

            self.floatingFoot_roll_state_observe = self.leftFoot_roll_state
            self.floatingFoot_pitch_state_observe = self.leftFoot_pitch_state
            self.groundFoot_roll_state_observe = self.rightFoot_roll_state
            self.groundFoot_pitch_state_observe = self.rightFoot_pitch_state

            self.x_dif = -self.x_dif

            self.floatingFoot_roll_torque = p.getJointState(self.robot, 22, physicsClientId=self.physicsClientId)[3]
            self.floatingFoot_pitch_torque = p.getJointState(self.robot, 21, physicsClientId=self.physicsClientId)[3]
            self.groundFoot_roll_torque = p.getJointState(self.robot, 6, physicsClientId=self.physicsClientId)[3]
            self.groundFoot_pitch_torque = p.getJointState(self.robot, 5, physicsClientId=self.physicsClientId)[3]

            self.walkStepCounter += 1
        if(self.walkStepCounter >= self.walk._stepPoint):
            self.walkStepCounter = 0
            if self.walkRightStep == False:
                self.walkRightStep = True
            else:
                self.walkRightStep = False

        if self.throwBallEnableList[self.terrainSwitch] == True:
            if (self.envStepCounter % (self.walk._stepPoint-1)) == 0:
                self.throwBall()
        self.envStepCounter += 1
        self.observation = self.compute_observation()
        # print(self._observation)
        reward = self.compute_reward()
        # print('reward : ', reward)
        done = self.compute_done()
        # print(done)

        if self.fall_down == True:
            reward -= 3

        return self.observation, reward, done, {}

    def compute_observation(self):
        _, cubeOrn = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.physicsClientId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn, physicsClientId=self.physicsClientId)
        # jointStateRR_torque = p.getJointState(self.robot, 6)[3]
        # jointStateRP_torque = p.getJointState(self.robot, 5)[3]
        # jointStateLR_torque = p.getJointState(self.robot, 22)[3]
        # jointStateLP_torque = p.getJointState(self.robot, 21)[3]
        # print([cubeEuler[0], cubeEuler[1], jointStateRR, jointStateRP, jointStateLR, jointStateLP])
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
        _, cubeOrn = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.physicsClientId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn, physicsClientId=self.physicsClientId)
        # print(math.sqrt(cubeEuler[0]*cubeEuler[0]+cubeEuler[1]*cubeEuler[1]))
        jointStateRR_torque = p.getJointState(self.robot, 6, physicsClientId=self.physicsClientId)[3]
        jointStateRP_torque = p.getJointState(self.robot, 5, physicsClientId=self.physicsClientId)[3]
        jointStateLR_torque = p.getJointState(self.robot, 22, physicsClientId=self.physicsClientId)[3]
        jointStateLP_torque = p.getJointState(self.robot, 21, physicsClientId=self.physicsClientId)[3]

        reward = 0.5-(cubeEuler[0]*cubeEuler[0]+cubeEuler[1]*cubeEuler[1])*1.0-abs(self.pos_z_plus) * 0.01 - \
            (abs(jointStateRR_torque)+abs(jointStateRP_torque) + abs(jointStateLR_torque)+abs(jointStateLP_torque))*0.1

        return reward

    def compute_done(self):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.physicsClientId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn, physicsClientId=self.physicsClientId)
        if (cubeEuler[0]*cubeEuler[0]+cubeEuler[1]*cubeEuler[1]) > self.fall_criteria:
            self.fall_down = True
            # return True
        else:
            self.fall_down = False
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

        orientation = p.getQuaternionFromEuler([0, 0, 0])

        if self.reset_physicsWorld_when_reset == True:
            if self.setLoadTerrain:
                p.resetSimulation(physicsClientId=self.physicsClientId)
                p.setTimeStep(self.timeStep, physicsClientId=self.physicsClientId)
                p.setPhysicsEngineParameter(numSolverIterations=self.numSolverIterations,
                                            physicsClientId=self.physicsClientId)
                self.robot = p.loadURDF(self.urdf_robot_path, [0, 0, 0.2],
                                        useFixedBase=False, physicsClientId=self.physicsClientId)
                self.motors.setRobot(self.robot, physicsClientId=self.physicsClientId)
                # self.ball = p.loadURDF(os.path.join(os.path.abspath(os.path.dirname(__file__)), "ball.xml"), [0, 15, 0.01])
                # plane = p.loadURDF("plane.urdf")
                self.loadTerrain()
        else:
            if self.setLoadTerrain:
                p.setGravity(0, 0, 0, physicsClientId=self.physicsClientId)
                p.resetBasePositionAndOrientation(
                    self.robot, [0, 0, 0.2], orientation, physicsClientId=self.physicsClientId)
                self.inverseKinematicsMove(
                    self.walk.walkPointStartLeftstepRightLeg[:, 0], self.walk.walkPointStartLeftstepLeftLeg[:, 0], 0, 0.2, 0.3)
                p.removeBody(self.terrain, physicsClientId=self.physicsClientId)
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

        p.setGravity(0, 0, 0, physicsClientId=self.physicsClientId)
        # p.resetBasePositionAndOrientation(self.robot, [0, 0, 0.2], orientation)
        self.inverseKinematicsMove(
            self.walk.walkPointStartLeftstepRightLeg[:, 0], self.walk.walkPointStartLeftstepLeftLeg[:, 0], 0, 0.2, 0.3)

        p.resetBasePositionAndOrientation(
            self.robot, [0, 0, 0.15-0.001*self.walk.walkPointStartLeftstepRightLeg[:, 0][2]], orientation, physicsClientId=self.physicsClientId)

        # startPos = [0.1, -random.uniform(3.0, 3.0), 0.15]
        # p.resetBasePositionAndOrientation(self.robot, startPos, orientation)

        p.setGravity(0, 0, -1, physicsClientId=self.physicsClientId)
        self.restSimulation(0.2)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.physicsClientId)
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
                self.walk.walkPointStartLeftstepRightLeg[:, i], self.walk.walkPointStartLeftstepLeftLeg[:, i], 0, self.walk._stepTime, 0)

        self.walkRightStep = True

        self.observation = self.compute_observation()
        return np.array(self.observation)

    def setRender(self, rows=1, columns=1, fig=None):
        img = np.zeros([200, 320])
        if fig is None:
            self.pltFig = plt.figure()
            plt.ion()
            plt.show()
            a = self.pltFig.add_subplot(rows, columns, 1)
        else:
            self.pltFig = fig
            a = self.pltFig.add_subplot(rows, columns, self.envNum+1)

        self.image = plt.imshow(img, interpolation='none')
        a.set_title('env'+self.envNumStr)
        plt.pause(0.1)

    def stopRender(self):
        plt.ioff()

    def camera_reset(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=-30, cameraPitch=-20,
                                     cameraTargetPosition=pos, physicsClientId=self.physicsClientId)

    def render(self, mode='human', close=False, cameraFollow=True):
        if cameraFollow == True:
            self.camera_reset()
        # img = np.random.rand(200, 320)
        # image = plt.imshow(img, interpolation='none', animated=True, label="blah")
        # ax = plt.gca()

        # camTargetPos = [0, 0, 0]
        robotPos, _ = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.physicsClientId)
        camTargetPos = [robotPos[0], robotPos[1], 0.2]
        cameraUp = [0, 0, 1]
        cameraPos = [1, 1, 1]

        pitch = -10.0

        roll = 0
        yaw = 0
        upAxisIndex = 2
        camDistance = 1
        pixelWidth = 320
        pixelHeight = 200
        nearPlane = 0.01
        farPlane = 100

        fov = 60

        viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex)
        aspect = pixelWidth / pixelHeight
        projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
        img_arr = p.getCameraImage(pixelWidth, pixelHeight, viewMatrix, projectionMatrix, shadow=1, lightDirection=[
            1, 1, 1], renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.physicsClientId)

        rgb = img_arr[2]

        # self.pltFig.add_subplot(2, 4, self.envNum+1)
        self.image.set_data(rgb)
        plt.draw()
        plt.pause(0.1)

    def render_directly(self):
        robotPos, _ = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.physicsClientId)
        camTargetPos = [robotPos[0], robotPos[1], 0.2]
        cameraUp = [0, 0, 1]
        cameraPos = [1, 1, 1]

        pitch = -10.0

        roll = 0
        yaw = 0
        upAxisIndex = 2
        camDistance = 1
        pixelWidth = 320
        pixelHeight = 200
        nearPlane = 0.01
        farPlane = 100

        fov = 60

        viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex)
        aspect = pixelWidth / pixelHeight
        projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
        img_arr = p.getCameraImage(pixelWidth, pixelHeight, viewMatrix, projectionMatrix, shadow=1, lightDirection=[
            1, 1, 1], renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.physicsClientId)

        rgb = img_arr[2]
        plt.imshow(rgb)  # np_img_arr)
        plt.show()

    def getRender(self):
        robotPos, _ = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.physicsClientId)
        camTargetPos = [robotPos[0], robotPos[1], 0.2]
        cameraUp = [0, 0, 1]
        cameraPos = [1, 1, 1]

        pitch = -10.0

        roll = 0
        yaw = 0
        upAxisIndex = 2
        camDistance = 1
        pixelWidth = 320
        pixelHeight = 200
        nearPlane = 0.01
        farPlane = 100

        fov = 60

        viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex)
        aspect = pixelWidth / pixelHeight
        projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
        img_arr = p.getCameraImage(pixelWidth, pixelHeight, viewMatrix, projectionMatrix, shadow=1, lightDirection=[
            1, 1, 1], renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.physicsClientId)

        rgb = img_arr[2]
        return rgb, self.envNum

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
            self.walk.walkPointStartRightstepRightLeg[:, 0], self.walk.walkPointStartRightstepLeftLeg[:, 0], 0, self.walk._stepTime, 0)

        rightStep = True
        for i in range(self.walk._stepPoint):
            self.inverseKinematicsMove(
                self.walk.walkPointStartRightstepRightLeg[:, i], self.walk.walkPointStartRightstepLeftLeg[:, i], 0, self.walk._stepTime, 0)
        rightStep = False
        for _ in range(stepNum):
            if(rightStep):
                for i in range(self.walk._stepPoint):
                    self.inverseKinematicsMove(
                        self.walk.walkPointRightStepRightLeg[:, i], self.walk.walkPointRightStepLeftLeg[:, i], 0, self.walk._stepTime, 0)
                rightStep = False
            else:
                for i in range(self.walk._stepPoint):
                    self.inverseKinematicsMove(
                        self.walk.walkPointLeftStepRightLeg[:, i], self.walk.walkPointLeftStepLeftLeg[:, i], 0, self.walk._stepTime, 0)
                rightStep = True
        if rightStep == True:
            for i in range(self.walk._stepPoint):
                self.inverseKinematicsMove(
                    self.walk.walkPointEndRightstepRightLeg[:, i], self.walk.walkPointEndRightstepLeftLeg[:, i], 0, self.walk._stepTime, 0)
        else:
            for i in range(self.walk._stepPoint):
                self.inverseKinematicsMove(
                    self.walk.walkPointEndLeftstepRightLeg[:, i], self.walk.walkPointEndLeftstepLeftLeg[:, i], 0, self.walk._stepTime, 0)
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
        count = int(restTime/self.timeStep)
        for _ in range(count):
            p.stepSimulation(physicsClientId=self.physicsClientId)

    def throwBall(self):
        robotPos, robotOrn = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.physicsClientId)
        angle = random.uniform(0, 2)*math.pi
        throwHeight = random.uniform(0.1, 0.5)
        position = [robotPos[0] + math.cos(angle) * (self.throwBallDistance + random.uniform(-0.1, 0.1)),
                    robotPos[1] + math.sin(angle) * (self.throwBallDistance + random.uniform(-0.1, 0.1)),
                    robotPos[2] + throwHeight]
        p.resetBasePositionAndOrientation(self.ball,  position, p.getQuaternionFromEuler([
                                          0, 0, 0]), physicsClientId=self.physicsClientId)
        force = random.uniform(0.5, 1.0)*self.throwBallForce
        p.applyExternalForce(self.ball, -1, [-math.cos(angle)*force,
                                             -math.sin(angle)*force,
                                             (random.uniform((0.15-throwHeight)*self.throwBallDistance, (0.65-throwHeight)*self.throwBallDistance))*force], [0, 0, 0], p.LINK_FRAME, physicsClientId=self.physicsClientId)

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
                                  startPos, p.getQuaternionFromEuler([1.5707963, 0, 1.5707963]), globalScaling=0.025, physicsClientId=self.physicsClientId)
        p.changeDynamics(self.terrain, -1,
                         lateralFriction=1.0,
                         spinningFriction=0.8,
                         physicsClientId=self.physicsClientId)
