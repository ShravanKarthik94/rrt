from __future__ import division
import pybullet as p
import pybullet_data
import numpy as np
import time
import argparse
from collections import defaultdict 
import copy
import math
import random

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

'''
    Function to move a distance 'step' from the point p1, along the line joining pt1 to pt2.
    Parameters:
        pt1 is coordinates of point in 3D space
        pt2 is coordinates of point in 3D space
    Return:
        3D coordinates of point step distance from p1 along line joining pt1, pt2
'''
def move_node(pt1, pt2):
    ptdiff = pt2-pt1
    norm = np.linalg.norm(ptdiff)
    newadd = (step/norm) * ptdiff
    newpt = pt1 + newadd
    return newpt

'''
    Function to find the nearest node on the tree for a given point.
    Parameters:
        treeNodes is a list of points on the tree
        point is a random point in 3D space for which the nearest point on the tree is to be determined
    Return:
        Node on the tree which is closest to 'point'
'''
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

'''
    Function to build an adjacency matrix for a given list of vertices and edges
    Parameters:
        edges: List of edges
        vertices: List of vertices
    Return:
        nxn matrix representing the adjacency matrix where n is the number of vertices
'''
def buildAdjList(edges, vertices):
    adjMat = np.zeros((len(vertices), len(vertices)))
    for x in range(len(edges)):
        edge = edges[x]
        adjMat[vertices.index(edge[0]),vertices.index(edge[1])] = 1
        adjMat[vertices.index(edge[1]),vertices.index(edge[0])] = 1
    return adjMat

'''
    Function to find path from start to destination given adjacencyMatrix
    Parameters:
        s: Coordinates Starting point
        d: Destination point
        vertex: List of coordinates of nodes on the tree
        adjList: Adjacency matrix for the tree with nodes 'vertex'
        visited: Array to keep track of nodes visited
        currPath: List of coordinates of nodes visited in current path traversal
    Return:
        No ruturn, however, if path from source to destination is found, global object finalPath is updated
        with the list of nodes to reach from start to destination
'''
def findPath(s, d, vertex, adjList, visited, currPath):
    global finalPath
    if len(finalPath) > 0:
        return 
    idx = vertex.index(s)
    visited[idx] = 1
    currPath.append(s)
    if d == s:
        finalPath = copy.deepcopy(currPath)
        return
    else:
        for x in range(len(vertex)):
            if adjList[idx,x] == 1 and visited[x] == 0:
                findPath(vertex[x], d, vertex, adjList, visited, currPath)
    currPath.pop()
    visited[idx] = 0

'''
    Function to check if there is a common node between trees v1 and v2
    Returns Inf, Inf if no such node exists
    Common node is found if there is a point in v1, v2 which is at a distance <= step size from each-other

    Parameters:
        v1: List of nodes belonging to tree with root as starting point
        v2: List of nodes belonging to tree with root as goal point
    Return:
        n1: Coordinates of node in v1 which is at a distance <= step size from 'n2'
        n2: Coordinates of node in v1 which is at a distance <= step size from 'n1'
'''
def checkconnection(v1, v2):
    n1 = math.inf
    n2 = math.inf
    len1 = len(v1)
    len2 = len(v2)
    for x in range(len1):
        for y in range(len2):
            dist = np.linalg.norm(np.array(v1[x])-np.array(v2[y]))
            if(dist <= step):
                n1 = v1[x]
                n2 = v2[y]
                return n1, n2
    return n1, n2


'''
    Function to implement RRT
    Parameters:
        None, however, expects 'step', 'biasCount', 'startSearch' and 'endSearch' parameters to be setup
        'step': Step-size by which the tree must grow
        'biasCount': Number of iterations after which the random point selected is set as the destination
        'startSearch': Lower limit indicating integer for which random point must be selected
        'endSearch': Upper limit indicating integer for which random point must be selected
    Return:
        None, however, calls findPath which returns the path from start to destination
'''
def rrt():
    path = []
    edges = []
    worldcrd = []
    path.append(start_conf)
    worldcrd.append(start_position)
    cntr = 1
    gp = np.array(goal_conf)
    while True:
        xnew = np.random.uniform(low=startSearch, high=endSearch)
        ynew = np.random.uniform(low=startSearch, high=endSearch)
        znew = np.random.uniform(low=startSearch, high=endSearch)
        pt2 = np.array((xnew, ynew, znew))
        if cntr % biasCount == 0:
            pt2 = gp
        nearestPt = find_nn(path, pt2)
        if(np.linalg.norm(nearestPt - pt2) == 0.0):
            continue

        newpt = move_node(nearestPt, pt2)
        idxmtch = path.index(tuple(nearestPt.reshape(1,-1)[0]))
        
        if not collision_fn(newpt):
            path.append(tuple(newpt.reshape(1,-1)[0]))
            edges.append(((tuple(nearestPt.reshape(1,-1)[0])),tuple(newpt.reshape(1,-1)[0])))
            lsz = p.getLinkState(ur5, 3)[0]
            worldcrd.append(lsz)
            p.addUserDebugLine(lineFromXYZ = worldcrd[idxmtch], lineToXYZ=lsz, lineColorRGB=[0, 1, 0], lineWidth = 0.5, lifeTime =5)
            if np.linalg.norm(gp-newpt) < step:
                #print(newpt)
                #print(np.linalg.norm(gp-newpt))
                break
        cntr = cntr + 1
    print("Done",cntr,len(path), len(edges))
    adjList = buildAdjList(edges, path)
    visited = np.zeros(len(path))
    currPath = []
    findPath(start_conf, tuple(newpt.reshape(1,-1)[0]), path, adjList, visited, currPath)
    print("Done with everything")

'''
    Function to implement Bi-RRT
    Parameters:
        None, however, expects 'step', 'biasCount', 'startSearch' and 'endSearch' parameters to be setup
        'step': Step-size by which the tree must grow
        'biasCount': The number of iterations after which the random point selected is set as the destination for tree
            with root as starting point, and start for tree with root as destination point
        'startSearch': Lower limit indicating integer for which random point must be selected
        'endSearch': Upper limit indicating integer for which random point must be selected
    Return:
        None, however, calls findPath which returns the path from start to destination
'''
def birrt():
    path1 = []
    edges1 = []
    path2 = []
    edges2 = []
    worldcrd1 = []
    worldcrd2 = []
    worldcrd1.append(start_position)
    worldcrd2.append(goal_position)
    path1.append(start_conf)
    path2.append(goal_conf)
    cntr = 1
    gp = np.array(goal_conf)
    sp = np.array(start_conf)
    while True:
        xnew = np.random.uniform(low=startSearch, high=endSearch)
        ynew = np.random.uniform(low=startSearch, high=endSearch)
        znew = np.random.uniform(low=startSearch, high=endSearch)
        xnew2 = np.random.uniform(low=startSearch, high=endSearch)
        ynew2 = np.random.uniform(low=startSearch, high=endSearch)
        znew2 = np.random.uniform(low=startSearch, high=endSearch)
        pt2s = np.array((xnew, ynew, znew))
        pt2e = np.array((xnew2, ynew2, znew2))
        if cntr % biasCount == 0:
            pt2s = gp
            pt2e = sp
        nearestPt = find_nn(path1, pt2s)
        nearestPt2 = find_nn(path2, pt2e)
        idxmtch1 = path1.index(tuple(nearestPt.reshape(1,-1)[0]))
        idxmtch2 = path2.index(tuple(nearestPt2.reshape(1,-1)[0]))
        addp1 = 1
        addp2 = 1
       
        cntr = cntr + 1

        if(np.linalg.norm(nearestPt - pt2s) == 0.0):
            addp1 = 0
        if(np.linalg.norm(nearestPt2 - pt2e) == 0.0):
            addp2 = 0
       
        if addp1==1 :
            newpt = move_node(nearestPt, pt2s)
            if not collision_fn(newpt):
                lsz = p.getLinkState(ur5, 3)[0]
                worldcrd1.append(lsz)
                p.addUserDebugLine(lineFromXYZ = worldcrd1[idxmtch1], lineToXYZ=lsz, lineColorRGB=[0, 1, 0], lineWidth = 0.5, lifeTime =10)
                path1.append(tuple(newpt.reshape(1,-1)[0]))
                edges1.append(((tuple(nearestPt.reshape(1,-1)[0])),tuple(newpt.reshape(1,-1)[0])))
        
        if addp2==1 :
            newpt2 = move_node(nearestPt2, pt2e)
            if not collision_fn(newpt2):
                lsz = p.getLinkState(ur5, 3)[0]
                worldcrd2.append(lsz)
                p.addUserDebugLine(lineFromXYZ = worldcrd2[idxmtch2], lineToXYZ=lsz, lineColorRGB=[0, 0, 1], lineWidth = 0.5, lifeTime =10)
                path2.append(tuple(newpt2.reshape(1,-1)[0]))
                edges2.append(((tuple(nearestPt2.reshape(1,-1)[0])),tuple(newpt2.reshape(1,-1)[0])))
        
        if addp1  == 1 or addp2 == 1:
            nd1 , nd2 = checkconnection(path1, path2)
            if nd1 == math.inf or nd2 == math.inf:
                continue
            else:
                path1.append(nd2)
                edges1.append((nd1,nd2))
                for x in range(len(path2)):
                    if not (path2[x] == nd2):
                        path1.append(path2[x])
                    if x < len(edges2):
                        edges1.append(edges2[x])
                break
    
    adjList = buildAdjList(edges1, path1)
    visited = np.zeros(len(path1))
    currPath = []
    findPath(start_conf, goal_conf, path1, adjList, visited, currPath)

'''
    Function to implement Bi-RRT with smoothing of the path found using Bi-RRT
    Parameters:
        None, however, expects 'step', 'biasCount', 'startSearch' and 'endSearch' parameters to be setup
        'step': Step-size by which the tree must grow
        'biasCount': The number of iterations after which the random point selected is set as the destination for tree
            with root as starting point, and start for tree with root as destination point
        'startSearch': Lower limit indicating integer for which random point must be selected
        'endSearch': Upper limit indicating integer for which random point must be selected
    Return:
        None, however, calls findPath which returns the path from start to destination. Subsequently, it update 'finalPath'
            to produce a 'smoothed' route from source to destination for the path determined using bi-rrt
'''
def birrt_smoothing():
    birrt()
    global finalPath
    for outer in range(smoothCount):
        start = 0
        end = len(finalPath) - 1
        idx1 = 0
        idx2 = 0
        while (idx1 == idx2) and (idx1 - idx2 < 2):
            idx1 = random.randint(start,end)
            idx2 = random.randint(start, end)
        numHops = abs(idx1 - idx2)
        if idx1 > idx2:
            temp = idx1 
            idx1 = idx2
            idx2 = temp
        newPt = []
        start = finalPath[idx1]
        end = finalPath[idx2]
        add = 0
        newPt.append(start)
        while True:
            pt2 = move_node(np.array(start), np.array(end))
            if not collision_fn(pt2):
                newPt.append(tuple(pt2.reshape(1,-1)[0]))
                start = tuple(pt2.reshape(1,-1)[0])
            else:
                break
            if np.linalg.norm(np.array(end) - np.array(pt2)) < step:
                newPt.append(end)
                add = 1
                break
        if add == 1:
            newLen = len(newPt)
            st = idx1
            end = idx2 + 1
            
            if newLen < numHops:
                for x in range(newLen):
                    ed = st+x
                    finalPath[ed] = newPt[x]
                ed = ed + 1
                for x in range(ed, end):
                    finalPath.remove(finalPath[ed])

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
    
    #Parameters for the different methods
    step = 0.05
    biasPercentage = 5
    biasCount = int(100.0 / biasPercentage)
    finalPath = []
    smoothCount = 100
    startSearch = -1
    endSearch = 1

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
            birrt_smoothing()
            path_conf = finalPath
        else:
            # using birrt without smoothing
            birrt()
            path_conf = finalPath
    else:
        # using rrt
        rrt()
        path_conf = finalPath
        #print(path_conf)

    if path_conf is None:
        # pause here
        input("no collision-free path is found within the time budget, finish?")
    else:
        # execute the path
        path_marker = []
        firstTime = 1
        prev = (math.inf, math.inf, math.inf)
        nulltemp = (math.inf, math.inf, math.inf)
        while True:
            for q in path_conf:
                set_joint_positions(ur5, UR5_JOINT_INDICES, q)
                lsz = p.getLinkState(ur5, 3)[0]
                    
                if firstTime == 0:
                    time.sleep(0.5)
                else:
                    if not (prev == nulltemp):
                        path_marker.append(p.addUserDebugLine(lineFromXYZ = prev, lineToXYZ=lsz, lineColorRGB=[1, 0, 0], lineWidth = 1.0, lifeTime =0))
                    prev = lsz
                    time.sleep(0.05)
            firstTime = 0
