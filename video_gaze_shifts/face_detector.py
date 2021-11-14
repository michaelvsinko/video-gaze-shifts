import numpy as np
import torch.nn as nn
from mediapipe.python.solutions.face_mesh import FaceMesh
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYE, FACEMESH_RIGHT_EYE
from ptgaze.common.camera import Camera
from ptgaze.common.face import Face

from video_gaze_shifts.utils import convert_pt


class FaceDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = FaceMesh(max_num_faces=1)

    def forward(self, image: np.ndarray) -> Face:
        height, width = image.shape[:2]
        pred = self.model.process(image[:, :, ::-1]).multi_face_landmarks[0]

        if not pred:
            return []

        landmarks = np.array([(landmark.x * width, landmark.y * height) for landmark in pred.landmark])
        bbox = np.vstack([landmarks.min(axis=0), landmarks.max(axis=0)])
        bbox = np.round(bbox).astype(np.int32)

        return Face(bbox=bbox, landmarks=landmarks)

    def eyes_centers(self, face: Face, camera: Camera):
        leye_center, reye_center = face.leye.center, face.reye.center
        leye_center, reye_center = camera.project_points(points3d=np.vstack([leye_center, reye_center]))
        leye_center, reye_center = convert_pt(leye_center), convert_pt(reye_center)

        return leye_center, reye_center
