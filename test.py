import cv2
import numpy as np


class ObjectMask:
    def __init__(
            self,
            src_video_path: str,
            output_video_path: str,
            detection_engine: object,
            frame_rescale_ratio: float = 1
    ):
        self.src_video_path: str = src_video_path
        self.output_video_path: str = output_video_path
        self.detection_engine: object = detection_engine
        """
        A object detection object that locates the pedestrians and vehicles in the frame.
        We assume it has the function:
        def run(frame: np.ndarry) -> List[np.ndarry]
        """

        self.frame_rescale_ratio: float = frame_rescale_ratio
        """"The ratio of rescaling a frame"""

        self.output_video_writer: cv2.VideoWriter

    def iterate_video(self):
        """Iterate the video and mask each frame"""
        cap = cv2.VideoCapture(self.src_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        _, frame = cap.read()
        self.output_video_writer = self._init_video_writer(self.output_path, fps, frame.shape[1], frame.shape[0])

        while cap.isOpened():
            _, frame = cap.read()

            rescaled_frame = cv2.resize(
                frame,
                (
                    int(
                        frame.shape[1] * self.frame_rescale_ratio
                    ),
                    int(
                        frame.shape[0] * self.frame_rescale_ratio
                    )
                )
            )

            masked_frame = self.mask(rescaled_frame)
            self.output_video_writer.write(masked_frame)

    def _init_video_writer(self, output_path: str, output_fps: int, frame_width: int,
                           frame_height: int) -> cv2.VideoWriter:
        width, height = int(frame_width * self.frame_rescale_ratio), int(frame_height * self.frame_rescale_ratio)
        self.output_video_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'MJPG'),
            output_fps,
            (width, height)
        )
        return self.output_video_writer

    def mask(self, frame: np.ndarray) -> np.ndarray:
        pedestrians_pos, vehicles_pos = self.detection_engine.run(frame)
        masked_frame = self.mask_pedestrians(frame, pedestrians_pos)
        masked_frame = self.mask_vehicles(masked_frame, vehicles_pos)
        return masked_frame

    @staticmethod
    def mask_pedestrians(frame: np.ndarray, pedestrians_pos: np.ndarray) -> np.ndarray:
        for x1, y1, x2, y2 in pedestrians_pos:
            # Find the width and height of each pedestrian object
            dy = y2 - y1
            dx = x2 - x1

            # Use the heuristic estimation to locate the face position
            # You may modify the parameters based on the video dataset.
            x1 = int(x1 + dx * 0.25)
            x2 = int(x2 - dx * 0.25)
            y2 = int(y1 + dy * 0.2)

            # Uncomment this line to visualize the effect
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

            roi_color = frame[y1:y2, x1:x2]

            # Blur the colored image
            blur = cv2.GaussianBlur(roi_color, (5, 5), 0)

            # Insert ROI back into image
            frame[y1:y2, x1:x2] = blur

        return frame

    @staticmethod
    def mask_vehicles(frame: np.ndarray, vehicles_pos: np.ndarray) -> np.ndarray:
        vehicles_pos = vehicles_pos.astype(int)
        for vehicle_pos in vehicles_pos:
            car_pixels = frame[vehicle_pos[1]:vehicle_pos[3], vehicle_pos[0]:vehicle_pos[2]]

            # Rescale each vehicle pixels into a fixed size.
            FIXED_WIDTH = 400
            rescaled_height = int(FIXED_WIDTH * car_pixels.shape[0] / car_pixels.shape[1])

            img = cv2.resize(car_pixels, (FIXED_WIDTH, rescaled_height))
            height, width = img.shape[:2]

            height_ratio = img.shape[0] / car_pixels.shape[0]
            width_ratio = img.shape[1] / car_pixels.shape[1]

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            cannyimg = cv2.Canny(gray, gray.shape[0], gray.shape[1])

            kernel = np.ones((5, 5), np.uint8)
            closingimg = cv2.morphologyEx(cannyimg, cv2.MORPH_CLOSE, kernel)

            openingimg = cv2.morphologyEx(closingimg, cv2.MORPH_OPEN, kernel)

            kernel = np.ones((5, 5), np.uint8)
            openingimg = cv2.morphologyEx(openingimg, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(openingimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for c in contours:
                x1 = np.min(c[:, :, 0])
                y1 = np.min(c[:, :, 1])
                x2 = np.max(c[:, :, 0])
                y2 = np.max(c[:, :, 1])

                if x2 - x1 > width * 0.1 or y2 - y1 > height * 0.1:
                    continue

                x1 = int(x1 / width_ratio)
                x2 = int(x2 / width_ratio)
                y1 = int(y1 / height_ratio)
                y2 = int(y2 / height_ratio)

                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue

                roi = frame[bbox[1] + y1: bbox[1] + y2, bbox[0] + x1: bbox[0] + x2]

                blur = cv2.GaussianBlur(roi, (21, 21), 0)
                frame[bbox[1] + y1: bbox[1] + y2, bbox[0] + x1: bbox[0] + x2] = blur
        return frame

