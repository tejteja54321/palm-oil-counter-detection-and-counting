from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
from ultralytics import YOLO
import os
import subprocess
from werkzeug.utils import secure_filename

import cv2
import numpy as np
from werkzeug.utils import secure_filename

# Initialize Flask App
app = Flask(__name__)
app.secret_key = 'your_secret_key'


# Folder for uploaded images and results
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER


users = {}

model = YOLO(r'F:/TEJA/IP/PROJ_10/runs/detect/train3/weights/best.pt')



# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Home Page After Login
@app.route('/about')
def about():
    if 'user' in session:
        return render_template('about.html', user=session['user'])
    return redirect(url_for('login'))

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email in users and users[email] == password:
            session['user'] = email
            flash('Login successful!', 'success')
            return redirect(url_for('about'))
        else:
            flash('Invalid email or password!', 'danger')
    
    return render_template('login.html')

# Register Page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
       
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
        elif email in users:
            flash('Email is already registered!', 'warning')
        else:
            users[email] = password
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

# Detection Page After Login
@app.route('/image')
def image():
    if 'user' in session:
        return render_template('image.html', user=session['user'])
    return redirect(url_for('login'))



@app.route("/result")
def result():
    if 'user' in session:
        return render_template('result.html', user=session['user'])
    return redirect(url_for('login'))



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("Image1")
        return redirect(url_for('about'))

    file = request.files['file']
    if file.filename == '':
        print("Image2")
        return redirect(url_for('about'))

    if file:
        print("Image3")
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video formats
            result_video_path = process_video(file_path, filename)
            return render_template('image.html', 
                                   original_file=filename, 
                                   result_video=result_video_path)

        else:  # Image file
            results = model(file_path)
            result_img_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{filename}")
            annotated_frame = results[0].plot()
            
            # Object counting
            class_counts = results[0].boxes.cls.cpu().numpy()
            class_counts = {model.names[int(cls_id)]: list(class_counts).count(cls_id) for cls_id in set(class_counts)}
            total_count = sum(class_counts.values())

            # Annotate frame with counts
            y_offset = 50
            font_scale = 1.5
            font_thickness = 3
            for class_name, count in class_counts.items():
                text = f"{class_name}: {count}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                text_x, text_y = 10, y_offset
                cv2.rectangle(annotated_frame, 
                              (text_x, text_y - text_size[1] - 10), 
                              (text_x + text_size[0] + 10, text_y + 10), 
                              (0, 0, 0), -1)  # Black background
                cv2.putText(annotated_frame, text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
                y_offset += text_size[1] + 20
            cv2.putText(annotated_frame, f"Total: {total_count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
            
            cv2.imwrite(result_img_path, annotated_frame)
            return render_template('image.html', 
                                   original_file=filename, 
                                   result_file=f"result_{filename}")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/results/videos/<filename>')
def result_video(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename, mimetype='video/mp4')

def process_video(input_path, filename):
    # OpenCV Video Capture
    cap = cv2.VideoCapture(input_path)
    output_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{filename}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    # Temporary file for processed frames
    temp_output = os.path.join(app.config['RESULTS_FOLDER'], "temp_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Temporary OpenCV video
    out = cv2.VideoWriter(temp_output, fourcc, fps, frame_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO Processing: Annotate the frame with detections
        results = model(frame)
        annotated_frame = results[0].plot()
        
        # Object counting
        class_counts = results[0].boxes.cls.cpu().numpy()
        frame_class_counts = {model.names[int(cls_id)]: list(class_counts).count(cls_id) for cls_id in set(class_counts)}
        frame_total_count = sum(frame_class_counts.values())
        
        # Annotate frame with counts
        y_offset = 50
        font_scale = 1.5
        font_thickness = 3
        for class_name, count in frame_class_counts.items():
            text = f"{class_name}: {count}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x, text_y = 10, y_offset
            cv2.rectangle(annotated_frame, 
                          (text_x, text_y - text_size[1] - 10), 
                          (text_x + text_size[0] + 10, text_y + 10), 
                          (0, 0, 0), -1)  # Black background
            cv2.putText(annotated_frame, text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
            y_offset += text_size[1] + 20
        cv2.putText(annotated_frame, f"Total: {frame_total_count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
        
        out.write(annotated_frame)

    cap.release()
    out.release()

    # FFmpeg Encoding for Final Output
    ffmpeg_command = [
        "ffmpeg", "-y",  # Overwrite output if exists
        "-i", temp_output,   # Input temporary video
        "-c:v", "libx264",   # H.264 codec for encoding
        "-preset", "medium", # Encoding speed
        "-crf", "23",        # Quality (Lower is better)
        "-c:a", "aac",       # Audio codec
        "-strict", "experimental",
        output_path          # Final output video
    ]

    try:
        # Run FFmpeg to re-encode
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e}")
        raise

    # Remove temporary file
    os.remove(temp_output)

    return f"result_{filename}"  # Return relative path



# Live Video Page After Login
@app.route('/live_video')
def live_video():
    if 'user' in session:
        return render_template('live_video.html', user=session['user'])
    return redirect(url_for('login'))


@app.route('/live_upload', methods=['POST'])
def live_upload():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Save the uploaded video file
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(video_path)

    # Start the video stream with detections
    return Response(generate(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Run inference on the frame
        results = model(frame)

        # Annotate the frame with detection boxes and class counts
        annotated_frame = results[0].plot()

        # Convert the frame to JPEG and yield as response
        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        frame_data = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    cap.release()

# Performance Page After Login
@app.route('/performance')
def performance():
    if 'user' in session:
        return render_template('performance.html', user=session['user'])
    return redirect(url_for('login'))

# Charts Page After Login
@app.route('/charts')
def charts():
    if 'user' in session:
        return render_template('charts.html', user=session['user'])
    return redirect(url_for('login'))


# Logout Route
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out!', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
