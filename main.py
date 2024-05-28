import cv2
from ultralytics import YOLO
import numpy as np
from sort.sort import Sort
from utils import get_vehicle, write_csv, read_number_plate

vehicle_tracker = Sort()

results = {}

# Load models
vehicle_detector_model = YOLO('yolov8n.pt')
number_plate_detector_model = YOLO('license_plate_detector.pt')

vehicles = [2, 3, 5, 7]

# Load video
video = cv2.VideoCapture('./test_vedio.mp4')

# Check if video opened successfully
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Read frames
frame_no = -1 
returned_value = True

while returned_value:
    frame_no += 1
    returned_value, frame = video.read()

    if returned_value and frame_no < 20:  # Limiting to 10 frames for debugging, remove or adjust as needed
        results[frame_no] = {}
        # Detect vehicles
        vehicle_detections = vehicle_detector_model(frame)[0]
        vehicle_detections_bbox = []
        for vehicle_detection in vehicle_detections.boxes.data.tolist():
            vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, vehicle_score, class_id = vehicle_detection
            if int(class_id) in vehicles:
                vehicle_detections_bbox.append([vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, vehicle_score])

        # Track vehicles
        vehicle_ids = vehicle_tracker.update(np.asarray(vehicle_detections_bbox))

        # Detect number plates
        number_plates = number_plate_detector_model(frame)[0]
        for number_plate in number_plates.boxes.data.tolist():
            num_plate_x1, num_plate_y1, num_plate_x2, num_plate_y2, num_plate_score, class_id = number_plate

            # Assign number plate to vehicle
            x_vehicle1, y_vehicle1, x_vehicle2, y_vehicle2, vehicle_id = get_vehicle(number_plate, vehicle_ids)

            if vehicle_id != -1:
                # Crop number plate
                number_plate_crop = frame[int(num_plate_y1):int(num_plate_y2), int(num_plate_x1):int(num_plate_x2), :]

                # Process number plate
                number_plate_crop_gray = cv2.cvtColor(number_plate_crop, cv2.COLOR_BGR2GRAY)
                _, number_plate_crop_gray_threshold = cv2.threshold(number_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read number plate text
                number_plate_text, number_plate_text_confidence_score = read_number_plate(number_plate_crop_gray_threshold)

                if number_plate_text is not None:
                    results[frame_no][vehicle_id] = {
                        'vehicle': {'bbox': [x_vehicle1, y_vehicle1, x_vehicle2, y_vehicle2]},
                        'number_plate': {
                            'bbox': [num_plate_x1, num_plate_y1, num_plate_x2, num_plate_y2],
                            'text': number_plate_text,
                            'bbox_score': num_plate_score,
                            'text_score': number_plate_text_confidence_score
                        }
                    }

# Write results
write_csv(results, './Detected_vehicles.csv')

# Release video
video.release()
print("Processing complete. Results written to './Detected_vehicles.csv'")
