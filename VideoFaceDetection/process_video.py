from dataclasses import dataclass
from typing import List
import cv2
from tqdm import tqdm
import datetime
import importlib.resources


@dataclass
class Face:
    top: int
    left: int
    right: int
    bottom: int
    confidence: float

    def __iter__(self):
        return iter((self.top, self.left, self.right, self.bottom, self.confidence))


class FaceDetectorYN:
    def __init__(self, width, height):
        self.detector = cv2.FaceDetectorYN.create(
            str(importlib.resources.path('VideoFaceDetection',
                'face_detection_yunet_2022mar.onnx')),
            "",
            (width, height)
        )

    def detect(self, frame) -> List[Face]:
        result = []

        faces = self.detector.detect(frame)
        if faces[1] is not None:
            for face in faces[1]:
                top = int(face[0])
                left = int(face[1])
                bottom = top + int(face[2])
                right = left + int(face[3])
                conf = face[-1]

                result.append(Face(top, left, right, bottom, conf))

        return result


def process_video(video_file, output_video, output_text, output_codec, confidence_threshold):
    vid = cv2.VideoCapture(video_file)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    total_frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*output_codec)
    out_v = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    detector = FaceDetectorYN(width, height)

    faces_info = []

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterator, *args, **kwargs):
            return iterator

    for current_frame in tqdm(range(total_frame_count)):
        if not vid.isOpened():
            break

        ret, frame = vid.read()
        if not ret:
            break

        faces = detector.detect(frame)
        for (top, left, right, bottom, confidence) in faces:
            if confidence < confidence_threshold:
                continue

            cv2.rectangle(frame,
                          (top, left), (bottom, right),
                          (0, 255, 0), 10)

            faces_info.append((current_frame, top, left, right, bottom))

        out_v.write(frame)

    vid.release()
    out_v.release()

    with open(output_text, 'w') as f:
        for current_frame, top, left, right, bottom in faces_info:
            time_str = f'{datetime.timedelta(seconds=current_frame // fps)}.{current_frame % fps}'
            bbox_str = f'({top}, {left}), ({bottom}, {right})'
            f.write(f'{time_str}\t{bbox_str}\n')
