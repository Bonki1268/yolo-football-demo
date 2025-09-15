from ultralytics import YOLO
import supervision as sv


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

    def get_object_tracks(self, frames):

        detection = self.detect_frames(frames)

        track_data = {
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

            track_data["players"].append({})
            track_data["referees"].append({})
            track_data["ball"].append({})

            for frame_tracks in get_tracks:
                box = frame_tracks[0].tolist()
                class_id = frame_tracks[3]
                track_id = frame_tracks[4]

                # print("Bound Box:", box)
                # print("Class ID:", class_id)
                # print("Track_ID:", track_id)

                if class_id == object_names_inverse["player"]:
                    track_data["players"][frame_index][track_id] = {"box": box}

                if class_id == object_names_inverse["referee"]:
                    track_data["referees"][frame_index][track_id] = {
                        "box": box}

            for frame_tracks in get_tracks:
                box = frame_tracks[0].tolist()
                class_id = frame_tracks[3]

                if class_id == object_names_inverse["ball"]:
                    track_data["ball"][frame_index][1] = {"box": box}

            print(track_data)
            print(object_names)
            break
