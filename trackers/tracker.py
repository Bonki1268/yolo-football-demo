from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
from utils import get_center, get_width


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            # detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections_batch = self.model(
                frames[i:i+batch_size], conf=0.1, device="mps")
            detections += detections_batch

        return detections

    def get_object_tracks(self, frames, read_db=False, db_path=None):

        if read_db and db_path is not None and os.path.exists(db_path):
            with open(db_path, "rb") as file:
                track_datas = pickle.load(file)
            return track_datas

        detection = self.detect_frames(frames)

        track_datas = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_index, detection in enumerate(detection):
            # print(detection)
            sv_detection = sv.Detections.from_ultralytics(detection)
            object_names = detection.names
            object_names_inverse = {
                value: key for key, value in object_names.items()
            }
            for object_index, class_id in enumerate(sv_detection.class_id):
                if object_names[class_id] == "goalkeeper":
                    sv_detection.class_id[object_index] = object_names_inverse["player"]
            # 追蹤
            get_tracks = self.tracker.update_with_detections(sv_detection)

            track_datas["players"].append({})
            track_datas["referees"].append({})
            track_datas["ball"].append({})

            for frame_tracks in get_tracks:
                box = frame_tracks[0].tolist()
                class_id = frame_tracks[3]
                track_id = frame_tracks[4]

                # print("Bound Box:", box)
                # print("Class ID:", class_id)
                # print("Track_ID:", track_id)

                if class_id == object_names_inverse["player"]:
                    track_datas["players"][frame_index][track_id] = {
                        "box": box}

                if class_id == object_names_inverse["referee"]:
                    track_datas["referees"][frame_index][track_id] = {
                        "box": box}

            for frame_tracks in get_tracks:
                box = frame_tracks[0].tolist()
                class_id = frame_tracks[3]

                if class_id == object_names_inverse["ball"]:
                    track_datas["ball"][frame_index][1] = {"box": box}

            # print(track_datas)
            # print(object_names)
            # break

        if db_path is not None:
            with open(db_path, "wb") as file:
                pickle.dump(track_datas, file)

        return track_datas

    def draw_ellipse(self, frame, box, color, track_id=None):
        x_center, _ = get_center(box)
        y_center = int(box[3])
        width = int(get_width(box))

        cv2.ellipse(
            frame,
            center=(x_center, y_center),
            axes=(width, int(0.3*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=245,
            color=color,
            thickness=2,
            lineType=cv2.LINE_8
        )

        rect_width = 36
        rect_height = 16
        rect_x1 = x_center - rect_width//2
        rect_x2 = x_center + rect_width//2
        rect_y1 = y_center + rect_height//2
        rect_y2 = y_center + rect_height + rect_height//2

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(rect_x1), int(rect_y1)),
                (int(rect_x2), int(rect_y2)),
                color,
                cv2.FILLED
            )

            text_x1 = rect_x1 + 5
            text_y1 = rect_y2

            if 9 < track_id < 100:
                text_x1 += rect_width//2//2 - 5
            if track_id < 10:
                text_x1 += rect_width//2 - 8

            cv2.putText(
                frame,
                f"{track_id}",
                (int(text_x1), int(text_y1)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (11, 45, 100),
                2
            )

        return frame

    def add_annotations(self, frames, track_datas):
        output_frames = []
        for frame_index, frame in enumerate(frames):
            frame = frame.copy()

            players = track_datas["players"][frame_index]
            referees = track_datas["referees"][frame_index]
            ball = track_datas["ball"][frame_index]

            for track_id, player in players.items():
                frame = self.draw_ellipse(
                    frame, player["box"], (255, 255, 255), track_id)

            for _, referee in referees.items():
                frame = self.draw_ellipse(frame, referee["box"], (0, 255, 255))

            output_frames.append(frame)
        return output_frames
