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

if page == "Home Page":
    # Dataset Description Page Code
    st.title("Data Driven Animal Species Recognition and Poacher Detection")
    
    st.image("cover.png")

    st.write("""
    Monitoring of protected areas to curb illegal activities like poaching is a monumental task. Real-time data acquisition has become easier with advances in unmanned aerial vehicles (UAVs) and sensors like TIR cameras, which allow surveillance at night when poaching typically occurs. However, it is still a challenge to accurately and quickly process large amounts of the resulting TIR data. This project aims to harness deep learning technologies to enhance wildlife recognition and poaching detection, providing conservationists, wildlife organisations, and authorities with actionable, real-time data to significantly bolster conservation efforts
    """)
             
    st.write("""
    For any inquiries regarding this project, contact Cheng Yee Ern at Universiti Malaya (22004791@siswa.um.edu.my) under supervision of Dr Riyaz Ahamed (riyaz@um.edu.my).
    """)
    

    # Add any other details or charts to explain the dataset further

elif page == "Dataset Description":
    # Dataset Description Page Code
    st.title("Dataset Description")

    st.image("birdsai.png", use_column_width=True)

    st.header("Overview")
    st.write("""
    The Benchmarking IR Dataset for Surveillance with Aerial Intelligence (BIRDSAI, pronounced "bird's-eye") is a long-wave thermal infrared dataset containing nighttime images of animals and poachers in Southern Africa. The dataset allows for benchmarking of algorithms for automatic detection and tracking of humans and animals with both real and synthetic videos. 
    
    It is a public dataset used for the ICVGIP Visual Data Challenge sourced from Labeled Information Library of Alexandria: Biology and Conservation (LILA BC)
    """)

    st.header("Data Format")
    st.write("""
    The dataset is composed of a set of real aerial TIR images and a set of synthetic aerial TIR images (generated with AirSim). 
    - Poachers 
    - Elephant
    - Lion
    - Giraffe
    - Dog
    - Crocodile (Synthetic)
    - Hippo (Synthetic)
    - Zebra (Synthetic)
    - Rhino (Syenthetic)
    """)

    # Add any other details or charts to explain the dataset further

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
    import shutil
    from ultralytics import YOLO  # Assuming this is the correct import for YOLO model
    import pandas as pd
    import plotly.express as px
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    def preprocess_frame(frame):
        """Preprocess the frame with Canny edge detection and adaptive thresholding."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, 100, 200)
        adaptive_thresh = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        combined_frame = cv2.merge([gray_frame, edges, adaptive_thresh])
        return combined_frame

    def generate_heatmap(bbox_locations, frame_shape, scale_factor=0.8):
        """Generate a heatmap from the accumulated bounding box locations and scale it down."""
        heatmap = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)

        for x, y, w, h in bbox_locations:
            # Add bounding box center to heatmap
            center_x, center_y = int(x + w / 2), int(y + h / 2)
            heatmap[center_y, center_x] += 1

        # Normalize heatmap to range [0, 255]
        heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = np.uint8(heatmap)

        # Apply the blue-red color map (JET) to the heatmap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Invert the color map by flipping the color channels
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        heatmap_colored = heatmap_colored[::-1, :, :]  # Flip the rows to reverse the color map

        # Resize heatmap to a smaller size based on scale factor
        resized_heatmap = cv2.resize(heatmap_colored, (int(frame_shape[1] * scale_factor), int(frame_shape[0] * scale_factor)))

        return resized_heatmap


    st.title("Test Your Video")

    st.header("Model Used: YOLO11")
    st.write(""" 
    **Mean Average Precision (mAP):** 0.633  
    **Frame Per Second (FPS):** 5.44  
    **Number of Parameters:** 2.59M  
    """)

    # Map class IDs to species names
    species_map = {
        0: "human",
        1: "elephants",
        2: "lions",
        3: "giraffes",
    }

    uploaded_video = st.file_uploader("Upload a video for inference (AVI, MP4)", type=["avi", "mp4"])

    if uploaded_video is not None:
        temp_dir = tempfile.TemporaryDirectory()
        input_video_path = os.path.join(temp_dir.name, uploaded_video.name)
        with open(input_video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.success("Uploaded video successfully.")

        try:
            trained_model = YOLO('yolo_corrected.pt')
            cap = cv2.VideoCapture(input_video_path)
            output_video_path = os.path.join(temp_dir.name, "output_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = None

            max_predictions = {species: 0 for species in species_map.values()}
            total_confidences = {species: [] for species in species_map.values()}
            poacher_detected = False
            total_time = 0
            total_frames = 0

            data_for_dashboard = []

            # List to store bounding box locations for heatmap
            bbox_locations = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.time()
                preprocessed_frame = preprocess_frame(frame)
                results = trained_model(preprocessed_frame, imgsz=640, conf=0.25)
                frame_with_boxes = results[0].plot()

                frame_predictions = {species: 0 for species in species_map.values()}
                frame_confidences = {species: [] for species in species_map.values()}

                for result in results[0].boxes.data:
                    class_id = int(result[-1])
                    confidence = result[-2]
                    species = species_map.get(class_id, None)

                    if species:
                        frame_predictions[species] += 1
                        frame_confidences[species].append(confidence)
                        total_confidences[species].append(confidence)

                        if class_id == 0 and not poacher_detected:
                            st.warning("Alert: Poacher detected!")
                            poacher_detected = True

                        # Track the bounding box locations (x, y, width, height)
                        x1, y1, x2, y2 = map(int, result[:4])
                        bbox_locations.append((x1, y1, x2 - x1, y2 - y1))

                for species, count in frame_predictions.items():
                    max_predictions[species] = max(max_predictions[species], count)

                for species, confidences in frame_confidences.items():
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    data_for_dashboard.append({
                        "Frame": total_frames + 1,
                        "Species": species,
                        "Count": frame_predictions[species],
                        "Average Confidence": avg_confidence
                    })

                if out is None:
                    height, width, _ = frame_with_boxes.shape
                    out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

                out.write(frame_with_boxes)
                frame_time = time.time() - start_time
                total_time += frame_time
                total_frames += 1

            cap.release()
            if out:
                out.release()

            # Get the frame shape (height, width) from the current frame with bounding boxes
            frame_shape = frame_with_boxes.shape
            # Generate heatmap from bounding box locations
            heatmap = generate_heatmap(bbox_locations, frame_shape=frame_with_boxes.shape)

            # Display the heatmap as an image
            st.subheader("Bounding Box Heatmap")
            st.image(heatmap, caption="Bounding Box Heatmap", width=frame_shape[1] * 0.8) 

            display_video_path = os.path.join(os.getcwd(), "output_video.mp4")
            shutil.copy(output_video_path, display_video_path)

            st.download_button(
                label="Download Processed Video",
                data=open(display_video_path, "rb").read(),
                file_name="output_video.mp4",
                mime="video/mp4"
            )

            # Convert data to DataFrame
            df = pd.DataFrame(data_for_dashboard)

            # Confidence Graph
            st.markdown("### Confidence Levels")
            confidence_fig = px.line(
                df,
                x="Frame",
                y="Average Confidence",
                color="Species",
                title="Confidence Score Over Frames",
                markers=True
            )
            st.plotly_chart(confidence_fig)

            # Prediction Summary
            st.markdown("### Prediction Summary")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Total Frames Processed", total_frames)
                st.metric("Total Time (s)", round(total_time, 2))

            with col2:
                st.metric("Average FPS", round(total_frames / total_time, 2) if total_time > 0 else "N/A")

            st.markdown("### Max Predictions Per Species")
            max_predictions_df = pd.DataFrame(max_predictions.items(), columns=["Species", "Max Count"])
            st.dataframe(max_predictions_df)

        except Exception as e:
            st.error(f"An error occurred while processing the video: {e}")
