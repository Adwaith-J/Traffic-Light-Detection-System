import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from collections import Counter

# Traffic Light Detector Class (embedded)
class TrafficLightDetector:
    def __init__(self):
        # HSV color ranges for traffic lights
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
        
        return self.remove_overlaps(detections)
    
    def remove_overlaps(self, detections):
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

# Streamlit App
def main():
    st.set_page_config(page_title="Traffic Light Detection", page_icon="🚦", layout="wide")
    
    st.title("🚦 Traffic Light Detection System")
    st.markdown("### Computer Vision-based Traffic Light Detection")
    
    # Initialize detector
    detector = TrafficLightDetector()
    
    # Sidebar
    st.sidebar.header("Parameters")
    detector.min_area = st.sidebar.slider("Min Area", 100, 1000, 200)
    detector.max_area = st.sidebar.slider("Max Area", 5000, 30000, 15000)
    show_masks = st.sidebar.checkbox("Show Color Masks", False)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload an image with traffic lights",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_file is not None:
        # Load and process image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        if len(image_array.shape) == 3:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_array
        
        # Detect traffic lights
        with st.spinner("Detecting traffic lights..."):
            start_time = time.time()
            detections = detector.detect_traffic_lights(image_bgr)
            processing_time = time.time() - start_time
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image_array, use_column_width=True)
        
        with col2:
            st.subheader("Detection Results")
            if detections:
                result_image = detector.draw_detections(image_bgr.copy(), detections)
                result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, use_column_width=True)
            else:
                st.image(image_array, use_column_width=True)
                st.warning("No traffic lights detected")
        
        # Statistics
        st.subheader("Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Lights Found", len(detections))
        with col2:
            st.metric("Processing Time", f"{processing_time:.2f}s")
        with col3:
            if detections:
                avg_confidence = np.mean([d['confidence'] for d in detections])
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
            else:
                st.metric("Avg Confidence", "N/A")
        
        # Detection details
        if detections:
            st.subheader("Detection Details")
            for i, detection in enumerate(detections):
                st.write(f"**Light {i+1}:** {detection['color'].upper()} "
                        f"(Confidence: {detection['confidence']:.3f})")
    
    else:
        st.info("👆 Upload an image to get started!")
        st.markdown("""
        ### How to use:
        1. **Upload an image** containing traffic lights
        2. **Adjust parameters** in the sidebar if needed
        3. **View results** with bounding boxes and confidence scores
        4. **Try different images** to test various scenarios
        """)

if __name__ == "__main__":
    main()