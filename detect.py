import argparse

from VideoFaceDetection import process_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detect faces from video file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('video', type=str,
                        help='input video file')
    parser.add_argument('output_video', type=str,
                        help='output video file', default='output.avi', nargs='?')
    parser.add_argument('output_text', type=str,
                        help='faces location file', default='output.txt', nargs='?')
    parser.add_argument('--codec', type=str,
                        help='output video codec', default='XVID')
    parser.add_argument('--confidence', type=float,
                        help='confidence treshold', default=0.97)
    parser.add_argument('--backend', type=str,
                        help='backend method', choices=['FaceDetectorYN', 'FaceRecognition', 'Cascade'], default='FaceDetectorYN')
    args = parser.parse_args()
    process_video(args.video,
                  args.output_video,
                  args.output_text,
                  args.codec,
                  args.confidence,
                  args.backend)
