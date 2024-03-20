import streamlit as st
from ultralytics import YOLO
import supervision as sv
import tempfile


st.set_page_config(layout="wide")
st.title("ML app")
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")
with st.spinner("load model"):
    model = load_model()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
tracker =sv.ByteTrack()
trace_annotator = sv.TraceAnnotator()
line_zone_annatator = sv.LineZoneAnnotator()





def get_frame(file_path):
    generator = sv.get_video_frames_generator(file_path)
    frame = next(generator)
@st.cache_data
def callback(frame,index:int):
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update_with_detections(detections)
    labels = [
        f"# {tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for tracker_id,class_id,confidence in
        zip(detections.tracker_id,detections.class_id,detections.confidence)
    ]
    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame,detections=detections)
    annotated_frame = trace_annotator.annotate(scene=annotated_frame,detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame,detections=detections,labels=labels)
    return line_zone_annatator.annotate(annotated_frame,line_counter=line_zone)

file = st.file_uploader("upload video",type=["mp4"])
if file is not None:
    video_file = open(file.name,"rb")
    vide_byte = video_file.read()
    st.video(vide_byte)

    t_file = tempfile.NamedTemporaryFile(delete=False)
    t_file.write(file.read())

    video_info = sv.VideoInfo.from_video_path(t_file.name)
    st.text(video_info)

    x_start = st.number_input("type x start point",value=None,key=1)
    y_start = st.number_input("type y start point",value=None,key=2)
    x_end = st.number_input("type x end point",value=None,key=3)
    y_end = st.number_input("type y end point",value=None,key=4)
    
    if x_start is not None and y_start is not None and x_end is not None and y_end is not None:
        start = sv.Point(x_start,y_start)
        end = sv.Point(x_end,y_end)
        line_zone = sv.LineZone(start=start,end=end)
        t_result = tempfile.NamedTemporaryFile(delete=False)

        sv.process_video(source_path=t_file.name,
                     target_path="result.mp4",
                     callback=callback
                     
                     )
        result_video = open("result.mp4","rb")
        result_byte = result_video.read()
        st.video(result_video)