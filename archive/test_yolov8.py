import cv2
import time
from models.yolov8_model import YOLOv8Model

def main():
    # Initialize camera with optimized parameters
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Initialize YOLOv8 model
    model = YOLOv8Model()
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Get detections
        detections = model.detect(frame)
        
        # Draw detections
        frame = model.draw_detections(frame, detections)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count >= 30:  # Update FPS every 30 frames
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()
        
        # Display FPS on frame
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('YOLOv8 Test', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 