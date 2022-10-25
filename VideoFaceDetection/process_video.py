import cv2
from tqdm import tqdm
import datetime


def process_video(video_file, output_video, output_text, output_codec):
    vid = cv2.VideoCapture(video_file)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    total_frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*output_codec)
    out_v = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    detector = cv2.FaceDetectorYN.create(
        'face_detection_yunet_2022mar.onnx',
        "",
        (width, height)
    )

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
        if faces[1] is not None:
            for face in faces[1]:
                top = int(face[0])
                left = int(face[1])
                bottom = top + int(face[2])
                right = left + int(face[3])
                conf = int(face[-1])

                cv2.rectangle(frame,
                              (top, left), (bottom, right),
                              (0, 255, 0), 10)
                cv2.putText(frame,
                            str(conf), (top - 30, left),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))

                faces_info.append((current_frame, top, left, right, bottom))

        out_v.write(frame)

    vid.release()
    out_v.release()

    with open(output_text, 'w') as f:
        for current_frame, top, left, right, bottom in faces_info:
            time_str = f'{datetime.timedelta(seconds=current_frame // fps)}.{current_frame % fps}'
            bbox_str = f'({top}, {left}), ({bottom}, {right})'
            f.write(f'{time_str}\t{bbox_str}\n')
