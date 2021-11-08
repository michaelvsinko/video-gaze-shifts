from typing import Union

import cv2
from ptgaze.common.face import Face
from ptgaze.common.visualizer import Visualizer as BaseVisualizer


class Visualizer(BaseVisualizer):
    def draw_head_pose(self, face: Face):
        self.draw_model_axes(face, 0.05, lw=2)
        euler_angles = face.head_pose_rot.as_euler("XYZ", degrees=True)
        face.change_coordinate_system(euler_angles)

    def draw_face_landmarks(self, face: Face):
        self.draw_points(face.landmarks, color=(0, 255, 255), size=1)

    def draw_gaze_vector(self, face: Face):
        self.draw_3d_line(face.center, face.center + 0.05 * face.gaze_vector)

    def draw_face_bbox(self, face: Face):
        self.draw_bbox(face.bbox)

    def draw_pitch_yaw(self, pitch: Union[str, float, int], yaw: Union[str, float, int]):
        cv2.putText(
            img=self.image,
            text=f"pitch: {pitch}, yaw: {yaw}",
            org=(490, 365),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 10, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    def draw_tag(self, tag: str):
        cv2.putText(
            img=self.image,
            text=f"tag: {tag}",
            org=(490, 375),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 10, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    def draw_areas(self, tag: str):
        center = (int(560.0 + (624.0 - 560.0) / 2.0), int(380.0 + (428.0 - 380.0) / 2.0))
        radius = int((624.0 - 560.0) / 6.0)
        cv2.rectangle(self.image, (560, 380), (624, 428), color=(240, 240, 240), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(self.image, (560, 380 + 10), (624, 428 - 10), color=(240, 240, 240), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(self.image, (560, 428 - 10), (624, 380 + 10), color=(240, 240, 240), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(
            self.image,
            (int(560.0 + (624.0 - 560.0) / 2.0), 380),
            (int(560.0 + (624.0 - 560.0) / 2.0), 428),
            color=(240, 240, 240),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(self.image, center=center, radius=radius, color=(240, 240, 240), thickness=-1, lineType=cv2.LINE_AA)

        if tag == "center":
            cv2.putText(
                img=self.image,
                text="+",
                org=(center[0] - 5, center[1] + 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 150, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        elif tag == "right-top":
            cv2.putText(
                img=self.image,
                text="+",
                org=(560 + 10, 380 + 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 150, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        elif tag == "left-top":
            cv2.putText(
                img=self.image,
                text="+",
                org=(624 - 20, 380 + 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 150, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        elif tag == "right-center":
            cv2.putText(
                img=self.image,
                text="+",
                org=(560 + 5, center[1] + 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 150, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        elif tag == "left-center":
            cv2.putText(
                img=self.image,
                text="+",
                org=(624 - 15, center[1] + 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 150, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        elif tag == "right-bottom":
            cv2.putText(
                img=self.image,
                text="+",
                org=(560 + 10, 428 - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 150, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        elif tag == "left-bottom":
            cv2.putText(
                img=self.image,
                text="+",
                org=(624 - 20, 428 - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 150, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
