import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array
from datetime import datetime
import pandas as pd
from flask import Flask, render_template_string, Response, request, redirect, url_for
import sqlite3
import time

# Database setup
def setup_database():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    # Create students table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            class TEXT NOT NULL,
            year TEXT NOT NULL,
            face_encoding BLOB
        )
    ''')
    
    # Create attendance table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# CNN Model for Face Recognition
def build_face_recognition_model():
    model = models.Sequential()
    
    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and Dense layers for embedding
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))  # Final embedding size
    
    return model

# Face Detection using OpenCV's Haar Cascade
def detect_face(frame):
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    face_images = []
    face_locations = []
    
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_img = frame[y:y+h, x:x+w]
        # Resize to standard size
        face_img = cv2.resize(face_img, (128, 128))
        # Convert to RGB (model expects RGB)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # Convert to array and normalize
        face_img = img_to_array(face_img) / 255.0
        
        face_images.append(face_img)
        face_locations.append((x, y, w, h))
    
    return face_images, face_locations

# Siamese Network for Face Comparison
class SiameseNetwork(tf.keras.Model):
    def __init__(self, embedding_model):
        super(SiameseNetwork, self).__init__()
        self.embedding_model = embedding_model
        
    def call(self, inputs):
        # Extract the two input images
        img1, img2 = inputs
        
        # Get embeddings
        embedding1 = self.embedding_model(img1)
        embedding2 = self.embedding_model(img2)
        
        # Calculate Euclidean distance
        distance = tf.sqrt(tf.reduce_sum(tf.square(embedding1 - embedding2), axis=1))
        
        return distance

# Face Registration Function
def register_new_face(name, class_name, year):
    cap = cv2.VideoCapture(0)
    face_data = []
    
    print("Please look at the camera and follow instructions...")
    poses = ["straight", "right", "left"]
    
    for pose in poses:
        print(f"Please look {pose} and press SPACE to capture...")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Display instructions
            cv2.putText(frame, f"Look {pose} and press SPACE to capture", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            # Show frame
            cv2.imshow("Registration", frame)
            
            # Wait for SPACE key
            key = cv2.waitKey(1)
            if key == 32:  # SPACE key
                face_images, _ = detect_face(frame)
                if face_images:
                    face_data.append(face_images[0])
                    print(f"Captured face looking {pose}")
                    break
                else:
                    print("No face detected, please try again")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Convert face data to numpy array
    face_data = np.array(face_data)
    
    # Save to database
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO students (name, class, year, face_encoding)
        VALUES (?, ?, ?, ?)
    ''', (name, class_name, year, face_data.tobytes()))
    
    student_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    print(f"Successfully registered {name} with ID {student_id}")
    return student_id

# Face Recognition and Attendance Marking
def recognize_face_and_mark_attendance(frame, siamese_model, embedding_model, threshold=0.5):
    face_images, face_locations = detect_face(frame)
    
    if not face_images:
        return frame, None
    
    # Get all student data
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, class, year, face_encoding FROM students")
    students = cursor.fetchall()
    conn.close()
    
    recognized_students = []
    
    for i, face_img in enumerate(face_images):
        face_img_batch = np.expand_dims(face_img, axis=0)
        
        min_distance = float('inf')
        recognized_student = None
        
        for student in students:
            student_id, name, class_name, year, face_encoding_bytes = student
            
            # Convert bytes back to numpy array
            face_encodings = np.frombuffer(face_encoding_bytes, dtype=np.float32).reshape(-1, 128, 128, 3)
            
            # Calculate similarity for each stored face
            distances = []
            for stored_face in face_encodings:
                stored_face_batch = np.expand_dims(stored_face, axis=0)
                # Use the siamese network to compare
                distance = siamese_model([face_img_batch, stored_face_batch])
                distances.append(distance.numpy()[0])
            
            # Get minimum distance
            student_min_distance = min(distances)
            
            if student_min_distance < min_distance:
                min_distance = student_min_distance
                recognized_student = (student_id, name, class_name, year)
        
        # If distance is below threshold, mark attendance
        if min_distance < threshold and recognized_student:
            student_id, name, class_name, year = recognized_student
            recognized_students.append((name, class_name, year))
            
            # Draw rectangle around face
            x, y, w, h = face_locations[i]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add name text
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Mark attendance in database
            current_date = datetime.now().strftime('%Y-%m-%d')
            current_time = datetime.now().strftime('%H:%M:%S')
            
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            
            # Check if already marked for today
            cursor.execute('''
                SELECT id FROM attendance 
                WHERE student_id = ? AND date = ?
            ''', (student_id, current_date))
            
            if not cursor.fetchone():
                cursor.execute('''
                    INSERT INTO attendance (student_id, date, time, status)
                    VALUES (?, ?, ?, 'present')
                ''', (student_id, current_date, current_time))
                
                conn.commit()
                print(f"Marked attendance for {name} on {current_date}")
            
            conn.close()
        else:
            # Face not recognized
            x, y, w, h = face_locations[i]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    return frame, recognized_students

# HTML Templates 

# Main Index Template
INDEX_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Face Recognition Attendance System</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 0; 
            background: linear-gradient(135deg, #e0e8ff 0%, #d8d0f9 100%);
            min-height: 100vh;
        }
        h1, h2 { 
            color: #6c46e5; 
        }
        h1 {
            text-align: center;
            font-size: 2.2rem;
            margin-top: 40px;
            margin-bottom: 30px;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px;
        }
        .nav-container {
            background-color: white;
            border-radius: 15px;
            padding: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
        }
        .nav-btn { 
            display: inline-flex;
            align-items: center;
            padding: 10px 20px; 
            color: #666;
            text-decoration: none; 
            margin: 0 15px;
            border-radius: 8px;
            transition: all 0.3s;
            font-weight: 500;
        }
        .nav-btn:hover {
            background-color: #f5f5ff;
        }
        .nav-btn.active {
            background-color: #6c46e5;
            color: white;
        }
        .video-container { 
            margin-top: 20px;
            background-color: #151c2e;
            border-radius: 15px;
            overflow: hidden;
            position: relative;
        }
        .video-container img {
            width: 100%;
            display: block;
        }
        .live-indicator {
            position: absolute;
            top: 15px;
            right: 15px;
            background-color: rgba(0,0,0,0.5);
            padding: 5px 10px;
            border-radius: 4px;
            color: white;
            font-size: 0.8rem;
            display: flex;
            align-items: center;
        }
        .live-dot {
            width: 8px;
            height: 8px;
            background-color: #ff3e3e;
            border-radius: 50%;
            margin-right: 5px;
            display: inline-block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Face Recognition Attendance System</h1>
        
        <div class="nav-container">
            <a href="/" class="nav-btn active">
                <span>📷</span> Mark Attendance
            </a>
            <a href="/register" class="nav-btn">
                <span>👤</span> Register
            </a>
            <a href="/attendance" class="nav-btn">
                <span>📊</span> Reports
            </a>
            <a href="/users" class="nav-btn">
                <span>👥</span> Users
            </a>
        </div>
        
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" alt="Camera Feed">
            <div class="live-indicator">
                <span class="live-dot"></span> Live
            </div>
        </div>
    </div>
</body>
</html>
'''

# Register Template
REGISTER_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Register New Student - Face Recognition Attendance System</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 0; 
            background: linear-gradient(135deg, #e0e8ff 0%, #d8d0f9 100%);
            min-height: 100vh;
        }
        h1, h2 { 
            color: #6c46e5; 
        }
        h1 {
            text-align: center;
            font-size: 2.2rem;
            margin-top: 40px;
            margin-bottom: 30px;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px;
        }
        .nav-container {
            background-color: white;
            border-radius: 15px;
            padding: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
        }
        .nav-btn { 
            display: inline-flex;
            align-items: center;
            padding: 10px 20px; 
            color: #666;
            text-decoration: none; 
            margin: 0 15px;
            border-radius: 8px;
            transition: all 0.3s;
            font-weight: 500;
        }
        .nav-btn:hover {
            background-color: #f5f5ff;
        }
        .nav-btn.active {
            background-color: #6c46e5;
            color: white;
        }
        .form-container {
            background-color: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            max-width: 600px;
            margin: 0 auto;
        }
        .form-group { 
            margin-bottom: 20px; 
        }
        label { 
            display: block; 
            margin-bottom: 8px; 
            font-weight: 500;
            color: #444;
        }
        input[type="text"] { 
            width: 100%; 
            padding: 12px; 
            box-sizing: border-box; 
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #6c46e5;
            box-shadow: 0 0 0 2px rgba(108, 70, 229, 0.1);
        }
        .btn { 
            display: inline-block; 
            padding: 12px 24px; 
            border: none;
            border-radius: 8px; 
            cursor: pointer;
            font-size: 16px;
            font-weight: 500;
            text-decoration: none;
        }
        .btn-primary {
            background-color: #6c46e5; 
            color: white; 
        }
        .btn-secondary {
            background-color: #f5f5f5;
            color: #666;
        }
        .btn:hover {
            opacity: 0.9;
        }
        .buttons {
            display: flex;
            justify-content: space-between;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Face Recognition Attendance System</h1>
        
        <div class="nav-container">
            <a href="/" class="nav-btn">
                <span>📷</span> Mark Attendance
            </a>
            <a href="/register" class="nav-btn active">
                <span>👤</span> Register
            </a>
            <a href="/attendance" class="nav-btn">
                <span>📊</span> Reports
            </a>
            <a href="/users" class="nav-btn">
                <span>👥</span> Users
            </a>
        </div>
        
        <div class="form-container">
            <h2>Register New Student</h2>
            <form method="POST">
                <div class="form-group">
                    <label for="name">Name:</label>
                    <input type="text" id="name" name="name" required>
                </div>
                <div class="form-group">
                    <label for="class">Class:</label>
                    <input type="text" id="class" name="class" required>
                </div>
                <div class="form-group">
                    <label for="year">Year:</label>
                    <input type="text" id="year" name="year" required>
                </div>
                <div class="buttons">
                    <a href="/" class="btn btn-secondary">Cancel</a>
                    <button type="submit" class="btn btn-primary">Register & Capture Face</button>
                </div>
            </form>
        </div>
    </div>
</body>
</html>
'''

# Reports/Attendance Template
REPORT_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Attendance Records - Face Recognition Attendance System</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 0; 
            background: linear-gradient(135deg, #e0e8ff 0%, #d8d0f9 100%);
            min-height: 100vh;
        }
        h1, h2 { 
            color: #6c46e5; 
        }
        h1 {
            text-align: center;
            font-size: 2.2rem;
            margin-top: 40px;
            margin-bottom: 30px;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px;
        }
        .nav-container {
            background-color: white;
            border-radius: 15px;
            padding: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
        }
        .nav-btn { 
            display: inline-flex;
            align-items: center;
            padding: 10px 20px; 
            color: #666;
            text-decoration: none; 
            margin: 0 15px;
            border-radius: 8px;
            transition: all 0.3s;
            font-weight: 500;
        }
        .nav-btn:hover {
            background-color: #f5f5ff;
        }
        .nav-btn.active {
            background-color: #6c46e5;
            color: white;
        }
        .content-container {
            background-color: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        .date-picker { 
            margin: 20px 0; 
            display: flex;
            align-items: center;
        }
        .date-picker label {
            margin-right: 10px;
            font-weight: 500;
        }
        .date-picker input[type="date"] {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-right: 10px;
        }
        .date-picker button {
            padding: 10px 15px;
            background-color: #6c46e5;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 20px; 
        }
        th, td { 
            padding: 15px; 
            text-align: left; 
        }
        th { 
            background-color: #f8f8ff; 
            color: #6c46e5;
            font-weight: 600;
        }
        tr {
            border-bottom: 1px solid #eee;
        }
        tr:last-child {
            border-bottom: none;
        }
        td {
            color: #444;
        }
        .present { 
            color: #4CAF50; 
            font-weight: 500;
        }
        .absent { 
            color: #f44336; 
            font-weight: 500;
        }
        .btn { 
            display: inline-block; 
            padding: 12px 24px; 
            border-radius: 8px; 
            cursor: pointer;
            margin-top: 20px;
            font-size: 16px;
            font-weight: 500;
            text-decoration: none;
            background-color: #6c46e5; 
            color: white;
        }
        .btn:hover {
            opacity: 0.9;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Face Recognition Attendance System</h1>
        
        <div class="nav-container">
            <a href="/" class="nav-btn">
                <span>📷</span> Mark Attendance
            </a>
            <a href="/register" class="nav-btn">
                <span>👤</span> Register
            </a>
            <a href="/attendance" class="nav-btn active">
                <span>📊</span> Reports
            </a>
            <a href="/users" class="nav-btn">
                <span>👥</span> Users
            </a>
        </div>
        
        <div class="content-container">
            <h2>Attendance Records</h2>
            <div class="date-picker">
                <form method="GET">
                    <label for="date">Select Date:</label>
                    <input type="date" id="date" name="date" value="{{ date }}">
                    <button type="submit">View</button>
                </form>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Class</th>
                        <th>Year</th>
                        <th>Time</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for record in attendance %}
                    <tr>
                        <td>{{ record['name'] }}</td>
                        <td>{{ record['class'] }}</td>
                        <td>{{ record['year'] }}</td>
                        <td>{{ record['time'] if record['status'] == 'present' else '-' }}</td>
                        <td class="{{ record['status'] }}">{{ record['status'].capitalize() }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
'''

# Users Template
USERS_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Registered Students - Face Recognition Attendance System</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 0; 
            background: linear-gradient(135deg, #e0e8ff 0%, #d8d0f9 100%);
            min-height: 100vh;
        }
        h1, h2 { 
            color: #6c46e5; 
        }
        h1 {
            text-align: center;
            font-size: 2.2rem;
            margin-top: 40px;
            margin-bottom: 30px;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px;
        }
        .nav-container {
            background-color: white;
            border-radius: 15px;
            padding: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
        }
        .nav-btn { 
            display: inline-flex;
            align-items: center;
            padding: 10px 20px; 
            color: #666;
            text-decoration: none; 
            margin: 0 15px;
            border-radius: 8px;
            transition: all 0.3s;
            font-weight: 500;
        }
        .nav-btn:hover {
            background-color: #f5f5ff;
        }
        .nav-btn.active {
            background-color: #6c46e5;
            color: white;
        }
        .content-container {
            background-color: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 20px; 
        }
        th, td { 
            padding: 15px; 
            text-align: left; 
        }
        th { 
            background-color: #f8f8ff; 
            color: #6c46e5;
            font-weight: 600;
        }
        tr {
            border-bottom: 1px solid #eee;
        }
        tr:last-child {
            border-bottom: none;
        }
        td {
            color: #444;
        }
        .search-container {
            margin-bottom: 20px;
        }
        .search-container input {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-sizing: border-box;
            font-size: 16px;
        }
        .search-container input:focus {
            outline: none;
            border-color: #6c46e5;
            box-shadow: 0 0 0 2px rgba(108, 70, 229, 0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Face Recognition Attendance System</h1>
        
        <div class="nav-container">
            <a href="/" class="nav-btn">
                <span>📷</span> Mark Attendance
            </a>
            <a href="/register" class="nav-btn">
                <span>👤</span> Register
            </a>
            <a href="/attendance" class="nav-btn">
                <span>📊</span> Reports
            </a>
            <a href="/users" class="nav-btn active">
                <span>👥</span> Users
            </a>
        </div>
        
        <div class="content-container">
            <h2>Registered Students</h2>
            <div class="search-container">
                <input type="text" id="searchInput" placeholder="Search for students..." onkeyup="searchStudents()">
            </div>
            <table id="studentsTable">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Name</th>
                        <th>Class</th>
                        <th>Year</th>
                    </tr>
                </thead>
                <tbody>
                    {% for student in students %}
                    <tr>
                        <td>{{ student['id'] }}</td>
                        <td>{{ student['name'] }}</td>
                        <td>{{ student['class'] }}</td>
                        <td>{{ student['year'] }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        function searchStudents() {
            const input = document.getElementById('searchInput');
            const filter = input.value.toUpperCase();
            const table = document.getElementById('studentsTable');
            const tr = table.getElementsByTagName('tr');

            for (let i = 1; i < tr.length; i++) {
                let found = false;
                const td = tr[i].getElementsByTagName('td');
                
                for (let j = 0; j < td.length; j++) {
                    if (td[j]) {
                        const txtValue = td[j].textContent || td[j].innerText;
                        if (txtValue.toUpperCase().indexOf(filter) > -1) {
                            found = true;
                            break;
                        }
                    }
                }
                
                if (found) {
                    tr[i].style.display = '';
                } else {
                    tr[i].style.display = 'none';
                }
            }
        }
    </script>
</body>
</html>
'''

# Create Flask application
app = Flask(__name__)
setup_database()  # Call database setup right after creating the app

# Flask Routes
@app.route('/')
def index():
    return render_template_string(INDEX_TEMPLATE)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        class_name = request.form['class']
        year = request.form['year']
        
        register_new_face(name, class_name, year)
        return redirect(url_for('index'))
    
    return render_template_string(REGISTER_TEMPLATE)

@app.route('/attendance')
def attendance():
    # Get date for which to show attendance
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    # Get attendance data
    conn = sqlite3.connect('attendance.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT s.name, s.class, s.year, a.time, a.status
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE a.date = ?
    ''', (date,))
    
    attendance_data = cursor.fetchall()
    attendance_list = [dict(row) for row in attendance_data]
    
    # Get all students to mark absent ones
    cursor.execute('SELECT id, name, class, year FROM students')
    all_students = cursor.fetchall()
    
    # Mark students as absent if not in attendance
    present_student_ids = set()
    for row in attendance_data:
        present_student_ids.add(row['student_id'] if 'student_id' in row else None)
    
    for student in all_students:
        if student['id'] not in present_student_ids:
            # Check if already marked as absent
            cursor.execute('''
                SELECT id FROM attendance 
                WHERE student_id = ? AND date = ? AND status = 'absent'
            ''', (student['id'], date))
            
            if not cursor.fetchone():
                cursor.execute('''
                    INSERT INTO attendance (student_id, date, time, status)
                    VALUES (?, ?, ?, 'absent')
                ''', (student['id'], date, '00:00:00'))
                
                conn.commit()
    
    # Refresh the attendance data
    cursor.execute('''
        SELECT s.name, s.class, s.year, a.time, a.status
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE a.date = ?
    ''', (date,))
    
    attendance_data = cursor.fetchall()
    attendance_list = [dict(row) for row in attendance_data]
    conn.close()
    
    return render_template_string(REPORT_TEMPLATE, attendance=attendance_list, date=date)

@app.route('/users')
def users():
    # Get all registered students
    conn = sqlite3.connect('attendance.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, name, class, year FROM students')
    students = cursor.fetchall()
    student_list = [dict(row) for row in students]
    conn.close()
    
    return render_template_string(USERS_TEMPLATE, students=student_list)

def gen_frames():
    # Try using a different backend
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    
    # Load models with correct architecture
    embedding_model = build_face_recognition_model()
    siamese_model = SiameseNetwork(embedding_model)
    
    # Try to load weights if available
    try:
        if os.path.exists('face_model_weights.h5'):
            print("Loading existing model weights...")
            embedding_model.load_weights('face_model_weights.h5')
    except:
        print("Could not load model weights, using untrained model")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            time.sleep(0.1)  # Add delay to prevent rapid failure loops
            continue
        
        try:
            # Recognize faces and mark attendance
            processed_frame, _ = recognize_face_and_mark_attendance(frame, siamese_model, embedding_model)
            
            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Return a blank or error frame
            height, width = frame.shape[:2]
            error_frame = np.zeros((height, width, 3), np.uint8)
            cv2.putText(error_frame, "Error processing frame", (10, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.5)  # Add delay to prevent rapid error loops

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train_model')
def train_model():
    # Get all student data
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, face_encoding FROM students")
    students = cursor.fetchall()
    conn.close()
    
    if not students:
        return "No students registered yet. Please register students first."
    
    # Prepare training data
    X = []
    y = []
    
    for student_id, face_encoding_bytes in students:
        # Convert bytes back to numpy array
        face_encodings = np.frombuffer(face_encoding_bytes, dtype=np.float32).reshape(-1, 128, 128, 3)
        
        for face_encoding in face_encodings:
            X.append(face_encoding)
            y.append(student_id)
    
    X = np.array(X)
    y = np.array(y)
    
    # Create model
    model = build_face_recognition_model()
    
    # Add classification layer
    num_classes = len(set(y))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Train the model
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    
    # Save the model
    model.save_weights('face_model_weights.h5')
    
    return "Model trained successfully!"

if __name__ == '__main__':
    app.run(debug=True)