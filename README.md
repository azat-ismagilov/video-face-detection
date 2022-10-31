# Video face detection
[Example](example)

## Requirenments
- Python3
- OpenCV >= 4.5.4, opencv-python
- [optional] tqdm
- [optional] [face-recognition](https://github.com/ageitgey/face_recognition#installation) if you want to use ```--backend=FaceRecognition```

## How to use
```
python3 detect.py [-h] [--codec CODEC] [--confidence CONFIDENCE] [--backend {FaceDetectorYN,FaceRecognition,Cascade}] video [output_video] [output_text]

Detect faces from video file

positional arguments:
  video                 input video file
  output_video          output video file (default: output.avi)
  output_text           faces location file (default: output.txt)

options:
  -h, --help            show this help message and exit
  --codec CODEC         output video codec (default: XVID)
  --confidence CONFIDENCE
                        confidence treshold (default: 0.97)
  --backend {FaceDetectorYN,FaceRecognition,Cascade}
                        backend method (default: FaceDetectorYN)
```

## About backends
- Cascade: Slightly faster
- FaceDetectorYN: Default backend, fast and reliable
- FaceRecognition: A lot slower, but results are much better

## Info
Also, check out my similar project, automated face recognition system with badge detection for ICPC world finals: https://github.com/azat-ismagilov/icpc-faces
