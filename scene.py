import json
import os
import random

from colmap_reader import readColmapSceneInfo
from model import GaussianModel
from params import TrainingParams
from utils.camera_utils import camera_to_json, create_camera_from_camera_info


class Scene:

    def __init__(self, model: GaussianModel,  training_params: TrainingParams, shuffle=True):
        self.model_path = training_params.model_path

        if os.path.exists(os.path.join(training_params.source_path, "sparse")):
            scene_info = readColmapSceneInfo(training_params.source_path, training_params.images, False)
        else:
            assert False, "Could not recognize scene type!"

        with open(scene_info.ply_path, 'rb') as src_file, \
                open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
            dest_file.write(src_file.read())

        cams = []
        if scene_info.test_cameras:
            cams.extend(scene_info.test_cameras)
        if scene_info.train_cameras:
            cams.extend(scene_info.train_cameras)

        cams_jsons = []
        for id, cam in enumerate(cams):
            cams_jsons.append(camera_to_json(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(cams_jsons, file, indent=4)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.train_cameras = list(map(
            lambda item: create_camera_from_camera_info(item[0], item[1]), enumerate(scene_info.train_cameras)
        ))
        self.test_cameras = list(map(
            lambda item: create_camera_from_camera_info(item[0], item[1]), enumerate(scene_info.test_cameras)
        ))

        model.init_from_scene_info(scene_info)
