import json
from pathlib import Path
from typing import Literal, Optional, Union

import cv2
import numpy as np
import ptgaze
import torch
import torch.nn as nn
from ptgaze.common.camera import Camera
from ptgaze.common.face import Face
from ptgaze.common.face_model_mediapipe import FaceModelMediaPipe
from ptgaze.head_pose_estimation import HeadPoseNormalizer
from torchvision import transforms as T
from tqdm import tqdm

from video_gaze_shifts.blink_detector import get_blink_detector, is_blink
from video_gaze_shifts.blink_extraction_tf.blink_extraction import prepare_input
from video_gaze_shifts.face_detector import FaceDetector
from video_gaze_shifts.gaze_estimator import GazeEstimator
from video_gaze_shifts.shift_estimator import estimate_shift, tags_to_stats
from video_gaze_shifts.visualizer import Visualizer

BASE_SIZE = (224, 224)
BASE_TRANSFORMS = T.Compose(
    [
        T.Lambda(lambda x: cv2.resize(x, BASE_SIZE)),
        T.Lambda(lambda x: x[:, :, ::-1].copy()),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ],
)
BASE_BLINK_TRANSFORMS = T.Compose(
    [
        T.Lambda(lambda x: prepare_input(x)),
    ],
)


class GazeRunner(nn.Module):
    def __init__(
        self,
        mode: Union[Literal["train"], Literal["eval"]] = "eval",
        gaze_backbone_name: str = "resnet18",
        gaze_estimator_path: Optional[Union[str, None, Path]] = None,
        transform: T.Compose = BASE_TRANSFORMS,
        blink_detector: Optional[str] = "tf__vgg16",
        blink_weights_path: Optional[str] = f"{Path(__file__).parent}/blink_extraction_tf/eye_detect_vgg16.h5",
        blink_transform: T.Compose = BASE_BLINK_TRANSFORMS,
        blink_mode: str = "mean",
        blink_thr: float = 0.15,
        device: Union[int, str, torch.device] = "cpu",
        cache: bool = True,
    ):
        super().__init__()

        self.mode = mode
        self.device = torch.device(device)

        self.face_model = FaceModelMediaPipe()
        camera_params_path = Path(ptgaze.__file__).parent / "data/calib/sample_params.yaml"
        self.raw_camera = Camera(str(camera_params_path))
        camera_params_path = Path(ptgaze.__file__).parent / "data/normalized_camera_params/eth-xgaze.yaml"
        self.normalized_camera = Camera(str(camera_params_path))

        self.face_detector = FaceDetector()
        self.head_pose_normalizer = HeadPoseNormalizer(
            camera=self.raw_camera,
            normalized_camera=self.normalized_camera,
            normalized_distance=0.6,
        )

        self.transform = transform

        self.blink_detector = None
        self.blink_transform = None
        self.blink_mode = None
        self.blink_thr = None
        if blink_detector is not None:
            self.blink_detector = get_blink_detector(name=blink_detector, weights_path=blink_weights_path)
            self.blink_transform = blink_transform
            if blink_mode in {"mean"}:
                self.blink_mode = blink_mode
            else:
                raise NotImplementedError
            self.blink_thr = blink_thr

        self.gaze_estimator = GazeEstimator(
            backbone_name=gaze_backbone_name,
            model_path=gaze_estimator_path,
            device=device,
            cache=cache,
        )

    def forward(
        self,
        path: str,
        mode: Union[Literal["image"], Literal["video"]] = "video",
        output_path: Optional[Union[str, None, Path]] = None,
        visualize: bool = False,
    ):
        visualizer = None
        writer = None
        if visualize:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            visualizer = Visualizer(camera=self.raw_camera, center_point_index=self.face_model.NOSE_INDEX)
        if visualize and mode == "video":
            writer = cv2.VideoWriter(
                output_path.as_posix(),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                (self.raw_camera.width, self.raw_camera.height),
            )

        if mode == "image":
            image = cv2.imread(filename=path)
            if self.mode == "train":
                out = self.forward_image(image=image, visualizer=visualizer, writer=writer)
            else:
                with torch.no_grad():
                    out = self.forward_image(image=image, visualizer=visualizer, writer=writer)

            if visualizer:
                cv2.imwrite(str(output_path), visualizer.image)

            yield out
        elif mode == "video":
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.raw_camera.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.raw_camera.height)

            ok: bool
            image: np.ndarray
            with tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT)) as progress_bar:
                while True:
                    ok, image = cap.read()
                    if not ok:
                        break

                    if self.mode == "train":
                        yield self.forward_image(image=image, visualizer=visualizer, writer=writer)
                    else:
                        with torch.no_grad():
                            yield self.forward_image(image=image, visualizer=visualizer, writer=writer)
                    progress_bar.update()

            cap.release()

            if writer is not None:
                writer.release()
        else:
            raise NotImplementedError

    def forward_image(
        self,
        image: np.ndarray,
        visualizer: Optional[Visualizer] = None,
        writer: Optional[cv2.VideoWriter] = None,
    ):
        undistorted = cv2.undistort(
            src=image,
            cameraMatrix=self.raw_camera.camera_matrix,
            distCoeffs=self.raw_camera.dist_coefficients,
        )

        if visualizer is not None:
            visualizer.set_image(image.copy())

        face: Face = self.face_detector(undistorted)
        face = self.forward_face_model(face=face)

        blink = False
        if self.blink_detector is not None:
            leye, reye = self.face_detector.eyes_centers(face=face, camera=self.raw_camera)
            leye = image[leye[1] - 42 : leye[1] + 42, leye[0] - 42 : leye[0] + 42]
            reye = image[reye[1] - 42 : reye[1] + 42, reye[0] - 42 : reye[0] + 42]

            blink = is_blink(
                eyes=(leye, reye),
                model=self.blink_detector,
                transform=self.blink_transform,
                mode=self.blink_mode,
                thr=self.blink_thr,
            )

        self.head_pose_normalizer.normalize(image=image, eye_or_face=face)

        if not blink:
            normalized_gaze_angles = self.forward_gaze_model(image=face.normalized_image)
            face.normalized_gaze_angles = normalized_gaze_angles
            face.angle_to_vector()
            face.denormalize_gaze_vector()

            # pitch - наклон вперед-назад
            # yaw - наклон влево-вправо
            vec_pitch, vec_yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            face_pitch, face_yaw, _ = face.head_pose_rot.as_euler("XYZ", degrees=True)

            # pitch+ - верх, pitch- - низ
            # yaw+ - право, yaw- - лево (относительно лица, не камеры)
            pitch, yaw = vec_pitch + face_pitch, vec_yaw - face_yaw

            tag = estimate_shift(pitch=pitch, yaw=yaw, mode="circle", r=10.0)
        else:
            pitch, yaw = None, None
            tag = "blink"

        if visualizer is not None:
            visualizer.draw_face_bbox(face=face)
            visualizer.draw_head_pose(face=face)
            if not blink:
                visualizer.draw_gaze_vector(face=face)
                visualizer.draw_pitch_yaw(pitch=int(pitch), yaw=int(yaw))
            visualizer.draw_tag(tag=tag)
            visualizer.draw_areas(tag=tag)
        if visualizer is not None and writer is not None:
            writer.write(visualizer.image)

        return pitch, yaw, tag

    def forward_face_model(self, face: Face) -> Face:
        self.face_model.estimate_head_pose(face=face, camera=self.raw_camera)
        self.face_model.compute_3d_pose(face=face)
        self.face_model.compute_face_eye_centers(face=face, mode="ETH-XGaze")

        return face

    def forward_gaze_model(self, image: np.ndarray) -> tuple:
        image = self.transform(image).unsqueeze(0).to(self.device)
        return self.gaze_estimator(image=image).cpu().numpy()[0]


if __name__ == "__main__":
    runner = GazeRunner()

    tags = []
    for _pitch, _yaw, tag in runner.forward(
        path="./examples/input_3.mp4",
        mode="video",
        output_path="./examples/output_3.mp4",
        visualize=True,
    ):
        tags.append(tag)

    stats = tags_to_stats(tags=tags)

    with open("./examples/output_3.json", "w") as f:
        json.dump(stats, f, indent=4)
