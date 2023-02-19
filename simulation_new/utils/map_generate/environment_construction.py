import numpy as np
import matplotlib.image as mpimg
import pybullet_data
import pybullet as p

def constructHeightFieldDictFromImg(imgpath: str, height, cell_size=.1, threshold=.5):
    img_array = mpimg.imread(imgpath)
    gray_img = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            if gray_img[i][j] >= threshold:
                gray_img[i][j] = height
            else:
                gray_img[i][j] = 0.
    result = {}
    result['type'] = 'height_field'
    result['scale'] = [cell_size, cell_size, 1.]
    result['grid_size'] = [gray_img.shape[0], gray_img.shape[1]]
    result['height_data'] = gray_img
    return result
    
def SimEnvironmentConstruction(pclient, obstacles: dict={}):
    """
    Load obstacles into the pybullet simulation environment
    :param pclient: pybullet client
    :param obstacles: dict of obstacles, temporarily support:
        {type: "box", center(1x3 array), half_extend(1x3 array), yaw_rad, fixed(bool)}
        {type: "cylinder", center(1x3 array), radius, height, fixed(bool)}
        {type: "ball", center(1x3 array), radius, fixed(bool)}
        {type: "height_field", scale(1x3 array), grid_size(1x2 array), height_data(MxN)}(seperatly loaded)
    """

    if p.getNumBodies(physicsClientId=pclient) == 0:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=pclient)
    
    uid_list = []
    only_height_field = None
    for (_, param) in obstacles.items():
        if param['type'] == "box":
            direction_quat = p.getQuaternionFromEuler([0., 0., param['yaw_rad']])
            visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
                                                halfExtents=param['half_extend'],
                                                physicsClientId=pclient)
            collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                halfExtents=param['half_extend'],
                                                physicsClientId=pclient)
            uid = p.createMultiBody(baseMass=0 if param['fixed'] else 1.,
                                baseCollisionShapeIndex=collisionShapeId,
                                baseVisualShapeIndex=visualShapeId,
                                basePosition=param['center'],
                                baseOrientation=direction_quat,
                                useMaximalCoordinates=True,
                                physicsClientId=pclient)
        elif param['type'] == "cylinder":
            visualShapeId = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                                                radius=param['radius'],
                                                length=param['height'],
                                                physicsClientId=pclient)
            collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                                radius=param['radius'],
                                                height=param['height'],
                                                physicsClientId=pclient)
            uid = p.createMultiBody(baseMass=0 if param['fixed'] else 1.,
                                baseCollisionShapeIndex=collisionShapeId,
                                baseVisualShapeIndex=visualShapeId,
                                basePosition=param['center'],
                                useMaximalCoordinates=True,
                                physicsClientId=pclient)
        elif param['type'] == 'ball':
            visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                radius=param['radius'],
                                                physicsClientId=pclient)
            collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_SPHERE,
                                                radius=param['radius'],
                                                physicsClientId=pclient)
            uid = p.createMultiBody(baseMass=0 if param['fixed'] else 1.,
                                baseCollisionShapeIndex=collisionShapeId,
                                baseVisualShapeIndex=visualShapeId,
                                basePosition=param['center'],
                                useMaximalCoordinates=True,
                                physicsClientId=pclient)
        elif param['type'] == "height_field":
            data = np.array(param['height_data'])
            if only_height_field is None:
                terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                                    meshScale=param['scale'],
                                                    heightfieldTextureScaling=(data.shape[0]-1)/2,
                                                    heightfieldData=param['height_data'].flatten(order='F'),
                                                    numHeightfieldRows=data.shape[0],
                                                    numHeightfieldColumns=data.shape[1],
                                                    physicsClientId=pclient)
                only_height_field = terrainShape
            else:
                terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                                    meshScale=param['scale'],
                                                    heightfieldTextureScaling=(data.shape[0]-1)/2,
                                                    heightfieldData=param['height_data'],
                                                    numHeightfieldRows=data.shape[0],
                                                    numHeightfieldColumns=data.shape[1],
                                                    replaceHeightfieldIndex=only_height_field,
                                                    physicsClientId=pclient)
                continue
            uid = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=terrainShape,
                                physicsClientId=pclient)
            pos_bias = [data.shape[0]*param['scale'][0] / 2,
                        data.shape[1]*param['scale'][1] / 2,
                        0]
            p.resetBasePositionAndOrientation(uid, pos_bias, [0,0,0,1], physicsClientId=pclient)
            p.changeVisualShape(uid, -1, rgbaColor=[1,1,1,1], physicsClientId=pclient)
        else:
            print("Warning: undefined obstacle type! Loading nothing!")
            uid = -1
        uid_list.append(uid)
    return uid_list

if __name__ == '__main__':
    import os
    import yaml
    client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.8, physicsClientId=client)
    p.setRealTimeSimulation(1, physicsClientId=client)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'obstacles.yaml')
    obstacles = yaml.load(open(path, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    imgpath = os.path.abspath(os.path.dirname(__file__)) + '/pictures/test.png'
    obstacles['grid'] = constructHeightFieldDictFromImg(imgpath, 3., cell_size=0.1)
    SimEnvironmentConstruction(client, obstacles)
    while True:
        continue
