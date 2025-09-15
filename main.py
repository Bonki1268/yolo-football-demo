from utils import read_video, save_video
from trackers import Tracker


def main():
    video_frames = read_video(
        "/Users/sam/Documents/Project/basketball/yolo_demo/my-app/videos/football.mp4")

    tracker = Tracker(
        "/Users/sam/Documents/Project/basketball/yolo_demo/my-app/models/best.pt")

    tracker.get_object_tracks(video_frames)

    output_video = save_video(video_frames, "result.mp4")


if __name__ == "__main__":
    main()
