import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from collections import Counter

# Import your detector class
from traffic_light_detector import TrafficLightDetector

# Page configuration
st.set_page_config(
    page_title="Traffic Light Detection System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Header
    st.title("🚦 Traffic Light Detection System")
    st.markdown("### Real-time traffic light detection using computer vision")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # Detection parameters
    st.sidebar.subheader("Detection Parameters")
    min_area = st.sidebar.slider("Minimum Area", 100, 1000, 200, 50)
    max_area = st.sidebar.slider("Maximum Area", 5000, 30000, 15000, 1000)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.1)
    show_masks = st.sidebar.checkbox("Show Color Detection Masks", False)
    
    # Initialize detector with custom parameters
    detector = TrafficLightDetector()
    detector.min_area = min_area
    detector.max_area = max_area
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["📷 Image Detection", "🎥 Video Detection", "📊 About"])
    
    with tab1:
        image_detection_interface(detector, show_masks, confidence_threshold)
    
    with tab2:
        video_detection_interface(detector, show_masks, confidence_threshold)
    
    with tab3:
        about_interface()

def image_detection_interface(detector, show_masks, confidence_threshold):
    """Image detection interface"""
    st.header("Image Traffic Light Detection")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload an image containing traffic lights"
    )
    
    # Example images section
    st.subheader("Or try these example scenarios:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔴 Test Red Detection"):
            st.info("Upload an image with red traffic lights for testing")
    
    with col2:
        if st.button("🟡 Test Yellow Detection"):
            st.info("Upload an image with yellow traffic lights for testing")
    
    with col3:
        if st.button("🟢 Test Green Detection"):
            st.info("Upload an image with green traffic lights for testing")
    
    if uploaded_file is not None:
        # Process the uploaded image
        process_uploaded_image(uploaded_file, detector, show_masks, confidence_threshold)

def process_uploaded_image(uploaded_file, detector, show_masks, confidence_threshold):
    """Process and analyze uploaded image"""
    try:
        # Load image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_array
        
        # Display original image info
        height, width = image_bgr.shape[:2]
        st.success(f"✅ Image loaded successfully: {width}x{height} pixels")
        
        # Detection process
        with st.spinner("🔍 Detecting traffic lights..."):
            start_time = time.time()
            detections = detector.detect_traffic_lights(image_bgr)
            processing_time = time.time() - start_time
        
        # Results summary
        st.subheader("🎯 Detection Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Lights Found", len(detections))
        
        with col2:
            st.metric("Processing Time", f"{processing_time:.2f}s")
        
        with col3:
            red_count = sum(1 for d in detections if d['color'] == 'red')
            st.metric("Red Lights", red_count)
        
        with col4:
            green_count = sum(1 for d in detections if d['color'] == 'green')
            st.metric("Green Lights", green_count)
        
        # Visual results
        if detections:
            # Draw detections
            result_image = detector.draw_detections(image_bgr.copy(), detections)
            result_image = detector.add_detection_statistics(result_image, detections)
            result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            # Display images side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Original Image")
                st.image(image_array, use_column_width=True)
            
            with col2:
                st.subheader("🎯 Detection Results")
                st.image(result_rgb, use_column_width=True)
            
            # Detection details
            st.subheader("📝 Detailed Detection Information")
            
            for i, detection in enumerate(detections):
                color = detection['color']
                confidence = detection['confidence']
                bbox = detection['bbox']
                center = detection['center']
                
                # Color-coded expander
                color_emoji = {"red": "🔴", "yellow": "🟡", "green": "🟢"}
                
                with st.expander(f"{color_emoji.get(color, '⚪')} Light {i+1}: {color.upper()} (Confidence: {confidence:.3f})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Position:** ({bbox[0]}, {bbox[1]})")
                        st.write(f"**Size:** {bbox[2]} × {bbox[3]} pixels")
                    
                    with col2:
                        st.write(f"**Center:** ({center[0]}, {center[1]})")
                        st.write(f"**Confidence:** {confidence:.1%}")
            
            # Show masks if requested
            if show_masks:
                st.subheader("🎨 Color Detection Masks")
                
                hsv_frame = detector.preprocess_frame(image_bgr)
                
                # Create individual color masks
                col1, col2, col3, col4 = st.columns(4)
                
                color_info = [
                    ("red", "🔴 Red Mask", col1),
                    ("yellow", "🟡 Yellow Mask", col2), 
                    ("green", "🟢 Green Mask", col3)
                ]
                
                combined_mask = np.zeros_like(image_bgr)
                
                for color_name, title, col in color_info:
                    mask = detector.detect_color(hsv_frame, color_name)
                    color_map = {'red': [255, 0, 0], 'yellow': [255, 255, 0], 'green': [0, 255, 0]}
                    
                    # Individual mask display
                    mask_colored = np.zeros_like(image_bgr)
                    mask_colored[mask > 0] = color_map[color_name]
                    mask_rgb = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
                    
                    with col:
                        st.write(title)
                        st.image(mask_rgb, use_column_width=True)
                    
                    # Add to combined mask
                    combined_mask[mask > 0] = color_map[color_name]
                
                # Combined mask
                with col4:
                    st.write("🎨 Combined Masks")
                    combined_rgb = cv2.cvtColor(combined_mask, cv2.COLOR_BGR2RGB)
                    st.image(combined_rgb, use_column_width=True)
            
            # Download section
            st.subheader("💾 Download Results")
            
            # Convert result to bytes for download
            result_pil = Image.fromarray(result_rgb)
            
            # Create download button
            import io
            img_buffer = io.BytesIO()
            result_pil.save(img_buffer, format='PNG')
            
            st.download_button(
                label="📥 Download Annotated Image",
                data=img_buffer.getvalue(),
                file_name=f"traffic_detection_{int(time.time())}.png",
                mime="image/png"
            )
            
        else:
            st.warning("⚠️ No traffic lights detected in this image.")
            st.info("💡 **Tips for better detection:**")
            st.write("• Ensure the image has clear, bright traffic lights")
            st.write("• Try adjusting the detection parameters in the sidebar")
            st.write("• Make sure lights are not too small or too large in the image")
            
            # Still show original image
            st.subheader("📋 Original Image")
            st.image(image_array, use_column_width=True)
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.info("Please try uploading a different image or check the file format.")

def video_detection_interface(detector, show_masks, confidence_threshold):
    """Video detection interface"""
    st.header("Video Traffic Light Detection")
    st.info("📹 Upload a video file to detect traffic lights frame by frame")
    
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video containing traffic lights"
    )
    
    if uploaded_video is not None:
        process_uploaded_video(uploaded_video, detector, show_masks, confidence_threshold)
    else:
        st.markdown("""
        ### 🎥 Video Detection Features:
        - **Frame-by-frame analysis** of uploaded videos
        - **Batch processing** with progress tracking
        - **Sample frame extraction** showing key detections
        - **Statistical analysis** of detection patterns
        - **Performance metrics** and timing information
        
        ### 📁 Supported Formats:
        - MP4, AVI, MOV, MKV
        - Various codecs and resolutions
        - Both short clips and longer videos
        """)

def process_uploaded_video(uploaded_video, detector, show_masks, confidence_threshold):
    """Process uploaded video file"""
    try:
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(uploaded_video.read())
            temp_path = temp_file.name
        
        # Open video
        cap = cv2.VideoCapture(temp_path)
        
        if not cap.isOpened():
            st.error("❌ Could not open video file")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Display video info
        st.success(f"✅ Video loaded successfully")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Resolution", f"{width}×{height}")
        with col2:
            st.metric("Duration", f"{duration:.1f}s")
        with col3:
            st.metric("FPS", fps)
        with col4:
            st.metric("Total Frames", frame_count)
        
        # Processing options
        st.subheader("🎛️ Processing Options")
        
        col1, col2 = st.columns(2)
        with col1:
            sample_rate = st.slider("Sample Every N Frames", 1, max(1, frame_count // 20), max(1, frame_count // 10))
        with col2:
            max_samples = st.slider("Maximum Samples", 5, 50, 20)
        
        if st.button("🚀 Start Video Processing", type="primary"):
            process_video_frames(cap, detector, frame_count, sample_rate, max_samples, show_masks)
        
        cap.release()
        os.unlink(temp_path)
        
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")

def process_video_frames(cap, detector, frame_count, sample_rate, max_samples, show_masks):
    """Process video frames with progress tracking"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processed_frames = []
    all_detections = []
    frame_idx = 0
    samples_collected = 0
    
    start_time = time.time()
    
    while samples_collected < max_samples:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            # Process this frame
            status_text.text(f'Processing frame {frame_idx}/{frame_count} (Sample {samples_collected + 1}/{max_samples})')
            
            detections = detector.detect_traffic_lights(frame)
            
            if detections:  # Only keep frames with detections
                result_frame = detector.draw_detections(frame.copy(), detections)
                result_frame = detector.add_detection_statistics(result_frame, detections)
                result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                
                processed_frames.append({
                    'frame_idx': frame_idx,
                    'image': result_rgb,
                    'detections': detections,
                    'detection_count': len(detections)
                })
                
                all_detections.extend(detections)
                samples_collected += 1
        
        frame_idx += 1
        progress_bar.progress(min(frame_idx / frame_count, samples_collected / max_samples))
    
    processing_time = time.time() - start_time
    
    # Display results
    status_text.text("✅ Processing complete!")
    
    if processed_frames:
        st.subheader("📊 Video Analysis Results")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Frames Processed", len(processed_frames))
        with col2:
            st.metric("Total Detections", len(all_detections))
        with col3:
            st.metric("Processing Time", f"{processing_time:.1f}s")
        with col4:
            avg_detections = len(all_detections) / len(processed_frames)
            st.metric("Avg Detections/Frame", f"{avg_detections:.1f}")
        
        # Color distribution
        color_counts = Counter([d['color'] for d in all_detections])
        
        if color_counts:
            st.subheader("🎨 Color Distribution")
            col1, col2, col3 = st.columns(3)
            
            colors = ['red', 'yellow', 'green']
            color_emojis = ['🔴', '🟡', '🟢']
            columns = [col1, col2, col3]
            
            for color, emoji, col in zip(colors, color_emojis, columns):
                count = color_counts.get(color, 0)
                percentage = (count / len(all_detections)) * 100 if all_detections else 0
                with col:
                    st.metric(f"{emoji} {color.upper()}", f"{count} ({percentage:.1f}%)")
        
        # Sample frames display
        st.subheader("🖼️ Sample Processed Frames")
        
        # Display frames in a grid
        cols_per_row = 3
        for i in range(0, len(processed_frames), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j in range(cols_per_row):
                if i + j < len(processed_frames):
                    frame_data = processed_frames[i + j]
                    with cols[j]:
                        st.image(
                            frame_data['image'],
                            caption=f"Frame {frame_data['frame_idx']} - {frame_data['detection_count']} lights",
                            use_column_width=True
                        )
    else:
        st.warning("⚠️ No traffic lights detected in the sampled frames.")
        st.info("💡 Try adjusting the detection parameters or using a different video.")

def about_interface():
    """About and information interface"""
    st.header("📋 About Traffic Light Detection System")
    
    st.markdown("""
    ### 🎯 System Overview
    This application uses **Computer Vision** and **Machine Learning** techniques to automatically detect and classify traffic lights in images and videos.
    
    ### 🔬 Technical Details
    
    **Detection Method:**
    - **Color Space:** HSV (Hue, Saturation, Value) for robust color detection
    - **Segmentation:** Multi-range color thresholding for red, yellow, and green
    - **Validation:** Shape analysis using area, aspect ratio, and circularity
    - **Post-processing:** Overlap removal and confidence scoring
    
    **Color Ranges (HSV):**
    - 🔴 **Red:** 0-10° and 170-180° (handles hue wraparound)
    - 🟡 **Yellow:** 20-30°
    - 🟢 **Green:** 40-80°
    
    ### ⚙️ Adjustable Parameters
    
    **Detection Sensitivity:**
    - **Minimum/Maximum Area:** Controls size range of detected objects
    - **Confidence Threshold:** Minimum confidence score for valid detections
    - **Circularity:** Shape validation for circular traffic lights
    
    ### 📈 Performance Features
    
    ✅ **Real-time Processing:** Fast frame-by-frame analysis  
    ✅ **Batch Processing:** Handle multiple images/videos  
    ✅ **Visual Feedback:** Bounding boxes, confidence scores, masks  
    ✅ **Statistics:** Detection counts, processing times, color distribution  
    ✅ **Export Options:** Download annotated results  
    
    ### 🎮 Usage Tips
    
    **For Best Results:**
    - Use clear, well-lit images with visible traffic lights
    - Ensure traffic lights are not too small (>200 pixels area)
    - Avoid heavily blurred or low-contrast images
    - Try adjusting parameters for different lighting conditions
    
    **Common Use Cases:**
    - 🚗 Autonomous vehicle development
    - 📊 Traffic analysis and monitoring  
    - 🎓 Educational computer vision projects
    - 🔬 Research and development
    """)
    
    # System statistics
    st.subheader("💻 System Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Libraries Used:**
        - OpenCV for image processing
        - NumPy for numerical operations
        - Streamlit for web interface
        - PIL for image handling
        """)
    
    with col2:
        st.success("""
        **Features:**
        - Multi-format support (JPG, PNG, MP4, AVI)
        - Real-time parameter adjustment
        - Interactive web interface
        - Detailed analysis reports
        """)

if __name__ == "__main__":
    main()