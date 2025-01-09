import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import shutil

# Set page configuration
st.set_page_config(page_title="Data Driven Animal Species Recognition and Human Detection", layout="wide")

# Check if the 'selected_page' exists in session_state; if not, initialize it
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "Home Page"  # Default page

# Page navigation
page = st.sidebar.radio("Select Page", ["Home Page", "Dataset Description", "Exploratory Data Analysis", "Test Your Video"], index=["Home Page", "Dataset Description", "Exploratory Data Analysis", "Test Your Video"].index(st.session_state.selected_page))

import streamlit as st

if page == "Home Page":
    # Dataset Description Page Code
    st.title("Data-Driven Animal Species Recognition and Poacher Detection", anchor="home")
    
    # Add an image with a caption
    st.image("cover.png", use_column_width=True, caption="Monitoring Wildlife with AI")

    # Description with proper formatting and separation
    st.write("""
    Monitoring of protected areas to curb illegal activities like poaching is a monumental task. Real-time data acquisition has become easier 
    with advances in unmanned aerial vehicles (UAVs) and sensors like TIR cameras, which allow surveillance at night when poaching typically occurs.
    However, it is still a challenge to accurately and quickly process large amounts of the resulting TIR data. This project aims to harness 
    deep learning technologies to enhance wildlife recognition and poaching detection, providing conservationists, wildlife organizations, 
    and authorities with actionable, real-time data to significantly bolster conservation efforts.
    """)

    # Inquiries section
    st.write("""
    ### Inquiries:
    For any inquiries regarding this project, contact Cheng Yee Ern at **Universiti Malaya**  
    Email: [22004791@siswa.um.edu.my](mailto:22004791@siswa.um.edu.my)  
    Supervised by: Dr Riyaz Ahamed  
    Email: [riyaz@um.edu.my](mailto:riyaz@um.edu.my)
    """)


elif page == "Dataset Description":
    # Dataset Description Page Code
    st.title("Dataset Description")

    # Image with a caption
    st.image("birdsai.png", use_column_width=True, caption="BIRDSAI Dataset for Wildlife Surveillance")

    # Overview Section with clean formatting
    st.header("Overview")
    st.write("""
    The **Benchmarking IR Dataset for Surveillance with Aerial Intelligence (BIRDSAI, pronounced "bird's-eye")** is a long-wave thermal infrared dataset containing nighttime images of animals and poachers in Southern Africa. 
    This dataset enables the benchmarking of algorithms for the automatic detection and tracking of humans and animals in aerial surveillance footage. It includes both real and synthetic videos.
    
    This dataset is publicly available and was used for the **ICVGIP Visual Data Challenge**. It is sourced from the **Labeled Information Library of Alexandria: Biology and Conservation (LILA BC)**.
    """)

    # Data Format Section with a bulleted list
    st.header("Data Format")
    st.write("""
    The dataset consists of two types of aerial thermal infrared images:
    - Real aerial TIR images
    - Synthetic aerial TIR images (generated with AirSim)

    These images cover the following subjects:
    - Poachers
    - Elephant
    - Lion
    - Giraffe
    - Dog
    - Crocodile 
    - Hippo 
    - Zebra 
    - Rhino 
    """)


elif page == "Exploratory Data Analysis":
    # Dataset Description Page Code
    st.title("Exploratory Data Analysis")

    st.write("This project only apply the real image dataset to ensure a more accurate, reliable, and practical solution for conservation surveillance.")

    st.header("Class distribution")

    st.image("species_distribution.png", use_column_width=True)

    st.write("""
    This graph suggest class imbalance for the dataset. Elephants dominate the dataset with over 40,000 annotations, making them the most frequently annotated species. In contrast, lions have fewer than 5,000 annotations, representing the least annotated species.
    
    """)

    st.header("Bounding Box Area Distribution")

    st.image("boundingbox_area_distribution.png", use_column_width=True)

    st.write("""
    Most bounding boxes are concentrated in the left area of the graph, indicating that the majority of the bounding boxes are extremely small. This suggests that many of the objects in the dataset occupy a small portion of the image
    """)

    st.header("Bounding Box Area Distribution by Class")
    
    st.image("species_boundingbox_area_distribution.png", use_column_width=True)
    
    st.write("""
    Elephants exhibit the largest average bounding box area, likely reflecting their physical size relative to other species in the dataset. Conversely, lions and "unknown species" are associated with the smallest bounding box areas, which may suggest that these species are either smaller or farther from the camera, leading to reduced bounding box sizes.
    """
    )
    
    st.header("Bounding Box Location Heatmap")
    
    st.image("heatmap_location.png", use_column_width=True)
    
    st.write("""
    The bounding boxes are primarily concentrated in the lower center of the image frames. This spatial distribution might indicate a dataset bias, where most objects are captured in this specific region of the frame. This could be due to factors like camera placement or the natural movement patterns of the animals being studied.
    """
    )

    elif page == "Test Your Video":
    import time
    import streamlit as st
    import cv2
    import os
    import tempfile
    from ultralytics import YOLO
    import pandas as pd
    import plotly.express as px
    import numpy as np

    # Helper Functions
    def preprocess_frame(frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, 100, 200)
        adaptive_thresh = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        combined_frame = cv2.merge([gray_frame, edges, adaptive_thresh])
        return combined_frame

    def generate_heatmap(bbox_locations, frame_shape, scale_factor=0.8):
        heatmap = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)
        for x, y, w, h in bbox_locations:
            center_x, center_y = int(x + w / 2), int(y + h / 2)
            heatmap[center_y, center_x] += 1
        heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = np.uint8(heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        resized_heatmap = cv2.resize(heatmap_colored, (int(frame_shape[1] * scale_factor), int(frame_shape[0] * scale_factor)))
        return resized_heatmap

    # Dashboard Layout
    st.title("Test Your Video")

    # Metrics Section
    st.subheader("Model Metrics", anchor="metrics")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.metric("mAP", "0.633", help="Mean Average Precision")
    with metrics_col2:
        st.metric("FPS", "5.44", help="Frames Per Second")
    with metrics_col3:
        st.metric("Parameters", "2.59M", help="Model Parameters")

    uploaded_video = st.file_uploader("Choose a video file", type=["avi", "mp4"], label_visibility="collapsed")

    # If a video is uploaded, process it
    if uploaded_video is not None:
        # Check if we have already processed the video
        if 'processed_video_path' not in st.session_state:
            temp_dir = tempfile.TemporaryDirectory()
            input_video_path = os.path.join(temp_dir.name, uploaded_video.name)
            with open(input_video_path, "wb") as f:
                f.write(uploaded_video.read())

            st.success("Video uploaded successfully!")

            try:
                # Load YOLO model
                trained_model = YOLO('yolo_corrected.pt')

                # Create a directory to save processed videos
                output_dir = "processed_videos"
                os.makedirs(output_dir, exist_ok=True)

                # Define the persistent output path
                output_video_path = os.path.join(output_dir, "output_video.mp4")

                # Process video
                cap = cv2.VideoCapture(input_video_path)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = None

                max_predictions = {}
                total_confidences = {}
                bbox_locations = []
                total_frames = 0
                total_time = 0

                data_for_dashboard = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    start_time = time.time()
                    preprocessed_frame = preprocess_frame(frame)
                    results = trained_model(preprocessed_frame, imgsz=640, conf=0.25)
                    frame_with_boxes = results[0].plot()

                    frame_time = time.time() - start_time
                    total_time += frame_time
                    total_frames += 1

                    for result in results[0].boxes.data:
                        class_id = int(result[-1])
                        confidence = result[-2]
                        x1, y1, x2, y2 = map(int, result[:4])
                        bbox_locations.append((x1, y1, x2 - x1, y2 - y1))

                        species = class_id  # Replace with your mapping logic
                        data_for_dashboard.append({
                            "Frame": total_frames,
                            "Species": species,
                            "Confidence": confidence.item()
                        })

                    if out is None:
                        height, width, _ = frame_with_boxes.shape
                        out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

                    out.write(frame_with_boxes)

                cap.release()
                if out:
                    out.release()

                frame_shape = frame_with_boxes.shape
                heatmap = generate_heatmap(bbox_locations, frame_shape)

                # Store the output path in session state
                st.session_state.processed_video_path = output_video_path
                st.session_state.data_for_dashboard = data_for_dashboard
                st.session_state.frame_shape = frame_shape
                st.session_state.bbox_locations = bbox_locations
                st.session_state.total_frames = total_frames
                st.session_state.total_time = total_time

            except Exception as e:
                st.error(f"An error occurred while processing the video: {e}")

        # Display processed video and analysis
        if 'processed_video_path' in st.session_state:
            output_video_path = st.session_state.processed_video_path
            frame_shape = st.session_state.frame_shape
            bbox_locations = st.session_state.bbox_locations
            data_for_dashboard = st.session_state.data_for_dashboard
            total_frames = st.session_state.total_frames
            total_time = st.session_state.total_time

            heatmap = generate_heatmap(bbox_locations, frame_shape)

            # Dashboard Display
            st.header("Analysis Results")

            # Bounding Box and Confidence Levels Section
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Bounding Box Heatmap")
                st.image(heatmap, caption="Heatmap of Bounding Box Locations", use_column_width=True)

            with col2:
                st.subheader("Confidence Levels Over Frames")
                df = pd.DataFrame(data_for_dashboard)
                confidence_fig = px.line(
                    df,
                    x="Frame",
                    y="Confidence",
                    color="Species",
                    title="Confidence Levels",
                    labels={"Confidence": "Confidence Level", "Frame": "Frame Number"}
                )
                # Customize layout for better appearance
                confidence_fig.update_layout(
                    height=frame_shape[0] * 0.8,
                    title=dict(x=0.5, xanchor="center"),
                    plot_bgcolor="#f9f9f9",
                    paper_bgcolor="#f9f9f9",
                    font=dict(family="Arial", size=12, color="#333")
                )
                st.plotly_chart(confidence_fig, use_container_width=True)

            # Processing Metrics Section
            st.subheader("Processing Metrics", anchor="metrics")
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Total Frames", total_frames)
            with metrics_col2:
                st.metric("Total Time (s)", round(total_time, 2))

            # Add download button for the processed video
            with open(output_video_path, "rb") as f:
                video_bytes = f.read()
            st.download_button(
                label="Download Processed Video",
                data=video_bytes,
                file_name="output_video.mp4",
                mime="video/mp4"
            )
