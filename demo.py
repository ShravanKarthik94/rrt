from __future__ import division
import pybullet as p
import pybullet_data
import numpy as np
import time
import argparse
from collections import defaultdict 
import copy

UR5_JOINT_INDICES = [0, 1, 2]
stp_size = 0.005

def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        p.resetJointState(body, joint, value)


def draw_sphere_marker(position, radius, color):
   vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
   marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
   return marker_id


def remove_marker(marker_id):
   p.removeBody(marker_id)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--birrt', action='store_true', default=False)
    parser.add_argument('--smoothing', action='store_true', default=False)
    args = parser.parse_args()
    return args

def move_node(pt1, pt2):
    ptdiff = pt2-pt1
    norm = np.linalg.norm(ptdiff)
    newadd = (step/norm) * ptdiff
    newpt = pt1 + newadd
    return newpt

def find_nn(treeNodes, point):
    treeLen = len(treeNodes)
    minDist = 100000000000
    finalNode = None
    for x in range(treeLen):
        dist = np.linalg.norm(point - np.array(treeNodes[x]))
        if(dist < minDist):
            finalNode = np.array(treeNodes[x])
            minDist = dist
    return finalNode

def buildAdjList(edges, vertices):
    adjMat = np.zeros((len(vertices), len(vertices)))
    for x in range(len(edges)):
        edge = edges[x]
        #print(edge)
        #print(vertices)
        #print(x)
        adjMat[vertices.index(edge[0]),vertices.index(edge[1])] = 1
        adjMat[vertices.index(edge[1]),vertices.index(edge[0])] = 1
    print("Built adj mat")
    return adjMat

def findPath(s, d, vertex, adjList, visited, currPath):
    global finalPath
    if len(finalPath) > 0:
        return 
    idx = vertex.index(s)
    visited[idx] = 1
    currPath.append(s)
    if d == s:
        print(currPath)
        print("Found path!!")
        finalPath = copy.deepcopy(currPath)
        return
    else:
        for x in range(len(vertex)):
            if adjList[idx,x] == 1 and visited[x] == 0:
                findPath(vertex[x], d, vertex, adjList, visited, currPath)
    currPath.pop()
    visited[idx] = 0

def rrt():
    path = []
    edges = []
    path.append(start_conf)
    cntr = 1
    gp = np.array(goal_conf)
    while True:
        xnew = np.random.uniform(low=-1, high=1)
        ynew = np.random.uniform(low=-1, high=1)
        znew = np.random.uniform(low=-1, high=1)
        pt2 = np.array((xnew, ynew, znew))
        if cntr % biasCount == 0:
            pt2 = gp
        nearestPt = find_nn(path, pt2)
        if(np.linalg.norm(nearestPt - pt2) == 0.0):
            continue

        newpt = move_node(nearestPt, pt2)
        
        if not collision_fn(newpt):
            path.append(tuple(newpt.reshape(1,-1)[0]))
            edges.append(((tuple(nearestPt.reshape(1,-1)[0])),tuple(newpt.reshape(1,-1)[0])))
            if np.linalg.norm(gp-newpt) < step:
                print(newpt)
                print(np.linalg.norm(gp-newpt))
                break
        cntr = cntr + 1
    print("Done",cntr,len(path))
    adjList = buildAdjList(edges, path)
    visited = np.zeros(len(path))
    currPath = []
    findPath(start_conf, tuple(newpt.reshape(1,-1)[0]), path, adjList, visited, currPath)
    print("Done with everything")

def birrt():
    #################################################
    # TODO your code to implement the birrt algorithm
    #################################################
    pass


def birrt_smoothing():
    ################################################################
    # TODO your code to implement the birrt algorithm with smoothing
    ################################################################
    pass


if __name__ == "__main__":
    args = get_args()

    # set up simulator
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
    p.resetDebugVisualizerCamera(cameraDistance=1.400, cameraYaw=58.000, cameraPitch=-42.200, cameraTargetPosition=(0.0, 0.0, 0.0))

    # load objects
    plane = p.loadURDF("plane.urdf")
    ur5 = p.loadURDF('assets/ur5/ur5.urdf', basePosition=[0, 0, 0.02], useFixedBase=True)
    obstacle1 = p.loadURDF('assets/block.urdf',
                           basePosition=[1/4, 0, 1/2],
                           useFixedBase=True)
    obstacle2 = p.loadURDF('assets/block.urdf',
                           basePosition=[2/4, 0, 2/3],
                           useFixedBase=True)
    obstacles = [plane, obstacle1, obstacle2]

    # start and goal
    start_conf = (-0.813358794499552, -0.37120422397572495, -0.754454729356351)
    start_position = (0.3998897969722748, -0.3993956744670868, 0.6173484325408936)
    goal_conf = (0.7527214782907734, -0.6521867735052328, -0.4949270744967443)
    goal_position = (0.35317009687423706, 0.35294029116630554, 0.7246701717376709)
    goal_marker = draw_sphere_marker(position=goal_position, radius=0.02, color=[1, 0, 0, 1])
    set_joint_positions(ur5, UR5_JOINT_INDICES, start_conf)
    step = 0.05
    biasPercentage = 5
    biasCount = int(100 / biasPercentage)
    finalPath = []

    # place holder to save the solution path
    path_conf = None

    # get the collision checking function
    from collision_utils import get_collision_fn
    collision_fn = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                                       attachments=[], self_collisions=True,
                                       disabled_collisions=set())

    if args.birrt:
        if args.smoothing:
            # using birrt with smoothing
            path_conf = birrt_smoothing()
        else:
            # using birrt without smoothing
            path_conf = birrt()
    else:
        # using rrt
        rrt()
        path_conf = finalPath
        print(path_conf)

    if path_conf is None:
        # pause here
        input("no collision-free path is found within the time budget, finish?")
    else:
        ###############################################
        # TODO your code to highlight the solution path
        ###############################################

        # execute the path
        while True:
            for q in path_conf:
                set_joint_positions(ur5, UR5_JOINT_INDICES, q)
                time.sleep(0.5)
