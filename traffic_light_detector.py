import cv2
import numpy as np
import argparse
import time
from collections import Counter
import os

class TrafficLightDetector:
    def __init__(self):
        self.color_ranges = {
            'red': [
                (np.array([0, 120, 70]), np.array([10, 255, 255])),    
                (np.array([170, 120, 70]), np.array([180, 255, 255]))  
            ],
            'yellow': [
                (np.array([20, 100, 100]), np.array([30, 255, 255]))   
            ],
            'green': [
                (np.array([40, 50, 50]), np.array([80, 255, 255]))     
            ]
        }
        
        self.min_area = 200
        self.max_area = 15000
        self.min_circularity = 0.3
        self.aspect_ratio_range = (0.7, 1.4) 
        
        
        self.detection_history = []
        self.history_size = 5
        
    def preprocess_frame(self, frame):
       
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        return hsv
    
    def detect_color(self, hsv_frame, color_name):
        
        masks = []
        
        for lower, upper in self.color_ranges[color_name]:
            mask = cv2.inRange(hsv_frame, lower, upper)
            masks.append(mask)
        
        combined_mask = np.zeros_like(masks[0])
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask
    
    def validate_contour(self, contour):
        
        area = cv2.contourArea(contour)
        if area < self.min_area or area > self.max_area:
            return False
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        
        if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
            return False
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < self.min_circularity:
            return False
        
        return True
    
    def detect_traffic_lights(self, frame):
        
        hsv_frame = self.preprocess_frame(frame)
        detections = []
        
        for color_name in ['red', 'yellow', 'green']:
            mask = self.detect_color(hsv_frame, color_name)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if self.validate_contour(contour):
                    x, y, w, h = cv2.boundingRect(contour)
                    roi_mask = mask[y:y+h, x:x+w]
                    confidence = np.sum(roi_mask > 0) / (w * h)
                    
                    if confidence > 0.3:  
                        detections.append({
                            'color': color_name,
                            'bbox': (x, y, w, h),
                            'confidence': confidence,
                            'center': (x + w//2, y + h//2)
                        })
        
        filtered_detections = self.remove_overlaps(detections)
        
        return filtered_detections
    
    def remove_overlaps(self, detections):
        """
        Remove overlapping detections, keeping the one with highest confidence
        """
        if len(detections) <= 1:
            return detections
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for detection in detections:
            x1, y1, w1, h1 = detection['bbox']
            
            overlap = False
            for accepted in filtered:
                x2, y2, w2, h2 = accepted['bbox']
                
                ix1 = max(x1, x2)
                iy1 = max(y1, y2)
                ix2 = min(x1 + w1, x2 + w2)
                iy2 = min(y1 + h1, y2 + h2)
                
                if ix1 < ix2 and iy1 < iy2:
                    intersection_area = (ix2 - ix1) * (iy2 - iy1)
                    area1 = w1 * h1
                    area2 = w2 * h2
                    overlap_ratio = intersection_area / min(area1, area2)
                    
                    if overlap_ratio > 0.3:  
                        overlap = True
                        break
            
            if not overlap:
                filtered.append(detection)
        
        return filtered
    
    def draw_detections(self, frame, detections):
        
        color_map = {
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0)
        }
        
        for detection in detections:
            color_name = detection['color']
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_map[color_name], 2)
            
            label = f"{color_name.upper()} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color_map[color_name], -1)
            
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def add_detection_statistics(self, frame, detections):
        
        color_counts = Counter([d['color'] for d in detections])
        
        y_offset = 30
        cv2.putText(frame, f"Total Lights: {len(detections)}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for color, count in color_counts.items():
            y_offset += 25
            cv2.putText(frame, f"{color.upper()}: {count}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

def process_single_image(image_path, detector, show_masks=False):
    
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        print("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif")
        return
    
    height, width = image.shape[:2]
    print(f"Image loaded: {width}x{height}")
    
    detections = detector.detect_traffic_lights(image)
    
    print(f"Found {len(detections)} traffic lights:")
    for i, detection in enumerate(detections):
        color = detection['color']
        confidence = detection['confidence']
        bbox = detection['bbox']
        print(f"  {i+1}. {color.upper()} at ({bbox[0]}, {bbox[1]}) - Confidence: {confidence:.3f}")
    
    result_image = detector.draw_detections(image.copy(), detections)
    result_image = detector.add_detection_statistics(result_image, detections)
    
    if show_masks and detections:
        hsv_frame = detector.preprocess_frame(image)
        mask_display = np.zeros_like(image)
        
        for color_name in ['red', 'yellow', 'green']:
            mask = detector.detect_color(hsv_frame, color_name)
            color_map = {'red': [0, 0, 255], 'yellow': [0, 255, 255], 'green': [0, 255, 0]}
            mask_display[mask > 0] = color_map[color_name]
        
        combined = np.hstack((image, result_image, mask_display))
        cv2.imshow('Traffic Light Detection - Original | Result | Masks', combined)
    else:
        combined = np.hstack((image, result_image))
        cv2.imshow('Traffic Light Detection - Original | Result', combined)
    
    print("\nControls:")
    print("Press 's' to save result image")
    print("Press any other key to exit")
    
    key = cv2.waitKey(0)
    
    if key == ord('s'):
        filename = image_path.split('/')[-1].split('\\')[-1]  
        name, ext = filename.rsplit('.', 1)
        output_name = f"{name}_detected.{ext}"
        cv2.imwrite(output_name, result_image)
        print(f"Result saved as: {output_name}")
        
        if show_masks and detections:
            mask_output = f"{name}_masks.{ext}"
            cv2.imwrite(mask_output, mask_display)
            print(f"Masks saved as: {mask_output}")
    
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Traffic Light Detection System')
    parser.add_argument('--source', '-s', default='0', 
                       help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--output', '-o', default=None,
                       help='Output video file path (optional)')
    parser.add_argument('--show-masks', action='store_true',
                       help='Show color detection masks')
    
    args = parser.parse_args()
    
    detector = TrafficLightDetector()
    
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height} @ {fps}fps")
    
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    detection_log = []
    
    print("Starting traffic light detection...")
    print("Press 'q' to quit, 's' to save screenshot, 'm' to toggle mask view")
    
    show_masks = args.show_masks
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break
        
        frame_start = time.time()
        
        detections = detector.detect_traffic_lights(frame)
        
        detection_log.append({
            'frame': frame_count,
            'timestamp': time.time(),
            'detections': len(detections),
            'colors': [d['color'] for d in detections]
        })
        
        annotated_frame = detector.draw_detections(frame.copy(), detections)
        annotated_frame = detector.add_detection_statistics(annotated_frame, detections)
        
        processing_time = time.time() - frame_start
        fps_text = f"FPS: {1/processing_time:.1f}"
        cv2.putText(annotated_frame, fps_text, (width - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if show_masks and detections:
            hsv_frame = detector.preprocess_frame(frame)
            mask_display = np.zeros_like(frame)
            
            for color_name in ['red', 'yellow', 'green']:
                mask = detector.detect_color(hsv_frame, color_name)
                color_map = {'red': [0, 0, 255], 'yellow': [0, 255, 255], 'green': [0, 255, 0]}
                mask_display[mask > 0] = color_map[color_name]
            
            combined = np.hstack((annotated_frame, mask_display))
            cv2.imshow('Traffic Light Detection - Original | Masks', combined)
        else:
            cv2.imshow('Traffic Light Detection', annotated_frame)
        
        if out is not None:
            out.write(annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_name = f"traffic_light_detection_{frame_count:06d}.jpg"
            cv2.imwrite(screenshot_name, annotated_frame)
            print(f"Screenshot saved: {screenshot_name}")
        elif key == ord('m'):
            show_masks = not show_masks
            print(f"Mask view: {'ON' if show_masks else 'OFF'}")
        
        frame_count += 1
    
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    
    print(f"\nPerformance Statistics:")
    print(f"Total frames processed: {frame_count}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    
    total_detections = sum(log['detections'] for log in detection_log)
    frames_with_detections = sum(1 for log in detection_log if log['detections'] > 0)
    
    print(f"\nDetection Statistics:")
    print(f"Total detections: {total_detections}")
    print(f"Frames with detections: {frames_with_detections}/{frame_count}")
    print(f"Detection rate: {frames_with_detections/frame_count*100:.1f}%")
    
    all_colors = []
    for log in detection_log:
        all_colors.extend(log['colors'])
    
    if all_colors:
        color_dist = Counter(all_colors)
        print(f"Color distribution: {dict(color_dist)}")
    
    log_file = "detection_log.txt"
    with open(log_file, 'w') as f:
        f.write("Frame,Timestamp,Detections,Colors\n")
        for log in detection_log:
            f.write(f"{log['frame']},{log['timestamp']},{log['detections']},\"{','.join(log['colors'])}\"\n")
    print(f"Detection log saved to: {log_file}")

if __name__ == "__main__":
    main()