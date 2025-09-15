import cv2


def read_video(file_path):
    video_capture = cv2.VideoCapture(file_path)

    frames = []

    while True:
        isReturn, frame = video_capture.read()
        if not isReturn:
            break
        frames.append(frame)

    video_capture.release()

    return frames


def save_video(frames, save_path):
    if not frames:
        raise ValueError("frames is empty")

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    output_video = cv2.VideoWriter(save_path, fourcc, 24,
                                   (frames[0].shape[1], frames[0].shape[0]))

    for frame in frames:
        output_video.write(frame)

    output_video.release()
