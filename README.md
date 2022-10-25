# Video face detection

## Requirenments
- Python3
- OpenCV >= 4.5.4, python-opencv
- [optional] tqdm

## How to use
```
python3 detect.py [-h] [--codec CODEC] video [output_video] [output_text]

Detect faces from video file

positional arguments:
  video          input video file
  output_video   output video file (default: output.avi)
  output_text    faces location file (default: output.txt)

options:
  -h, --help     show this help message and exit
  --codec CODEC  output video codec (default: XVID)
```
