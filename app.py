from flask import Flask, render_template, request, redirect, session, jsonify
import sqlite3
import cv2
import numpy as np
import mediapipe as mp
import joblib
import base64

# prepare pose model globals
model = None
encoder = None
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose()


def load_models():
    global model, encoder
    if model is None or encoder is None:
        try:
            model = joblib.load("mediapipe_rf_model.pkl")
            encoder = joblib.load("mediapipe_label_encoder.pkl")
        except Exception as e:
            print("Failed to load pose models:", e)


def dataurl_to_image(data_url):
    header, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def extract_features(landmarks):
    hip = landmarks[23]
    features = []
    for lm in landmarks:
        features.extend([lm.x - hip.x, lm.y - hip.y, lm.z - hip.z])
    return np.array(features).reshape(1, -1)


def predict_pose_from_image(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose_detector.process(rgb)
    if not result.pose_landmarks:
        return None, 0.0
    landmarks = result.pose_landmarks.landmark
    feats = extract_features(landmarks)
    probs = model.predict_proba(feats)[0]
    max_prob = float(np.max(probs))
    if max_prob < 0.5:
        return "Incorrect Pose", max_prob
    pred = int(np.argmax(probs))
    pose_name = encoder.inverse_transform([pred])[0]
    return pose_name, max_prob

app = Flask(__name__)
app.secret_key = "yoga_secret"

# Initialize Database
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT,
            role TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            accuracy REAL,
            streak INTEGER,
            level TEXT,
            date_completed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()

init_db()

# Home
@app.route("/")
def home():
    return render_template("index.html")

# Register
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")
    
    name = request.form["name"]

    email = request.form["email"]
    password = request.form["password"]
    role = request.form["role"]

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    try:
        c.execute("INSERT INTO users (name,email,password,role) VALUES (?,?,?,?)",
                  (name,email,password,role))
        conn.commit()
    except:
        return "User already exists"

    conn.close()
    return redirect("/login")

# Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        role = request.form["role"]

        conn = sqlite3.connect("database.db")
        c = conn.cursor()

        c.execute("SELECT * FROM users WHERE email=? AND password=? AND role=?",
                  (email,password,role))

        user = c.fetchone()
        conn.close()

        if user:
            session["user"] = user[1]
            session["role"] = user[4]

            if role == "admin":
                return redirect("/admin")
            else:
                return redirect("/dashboard")

        return "Invalid Credentials"

    return render_template("login.html")

# User Dashboard
@app.route("/dashboard")
def dashboard():
    if "user" in session and session["role"] == "user":
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        
        # Get user ID
        c.execute("SELECT id FROM users WHERE name=?", (session["user"],))
        user_id = c.fetchone()[0]
        
        # Get total sessions
        c.execute("SELECT COUNT(*) FROM sessions WHERE user_id=?", (user_id,))
        total_sessions = c.fetchone()[0]
        
        # Get average accuracy
        c.execute("SELECT AVG(accuracy) FROM sessions WHERE user_id=?", (user_id,))
        avg_accuracy = c.fetchone()[0]
        if avg_accuracy is None:
            avg_accuracy = 0
        
        # Get current streak (maximum consecutive days with sessions)
        c.execute("""
            SELECT COUNT(DISTINCT DATE(date_completed)) as streak_days
            FROM sessions 
            WHERE user_id = ? 
            AND date_completed >= date('now', '-30 days')
        """, (user_id,))
        streak = c.fetchone()[0]
        if streak is None:
            streak = 0
        
        conn.close()
        
        return render_template("dashboard.html", 
                             name=session["user"],
                             sessions=total_sessions,
                             accuracy=round(avg_accuracy, 1),
                             streak=streak)
    return redirect("/login")

# Admin Dashboard
@app.route("/admin")
def admin():
    if "user" in session and session["role"] == "admin":
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        
        # Get total users
        c.execute("SELECT COUNT(*) FROM users WHERE role='user'")
        total_users = c.fetchone()[0]
        
        # Get active users (users with sessions)
        c.execute("SELECT COUNT(DISTINCT user_id) FROM sessions")
        active_users = c.fetchone()[0]
        
        # Get detailed user statistics
        c.execute("""
            SELECT 
                u.name,
                COUNT(s.id) as sessions,
                ROUND(AVG(s.accuracy), 1) as accuracy,
                MAX(s.date_completed) as last_activity
            FROM users u
            LEFT JOIN sessions s ON u.id = s.user_id
            WHERE u.role = 'user'
            GROUP BY u.id, u.name
            ORDER BY sessions DESC
        """)
        user_stats = []
        for row in c.fetchall():
            user_stats.append({
                'name': row[0],
                'sessions': row[1] or 0,
                'accuracy': row[2] or 0,
                'last_activity': row[3] or 'Never'
            })
        
        conn.close()
        
        return render_template("admin.html", 
                             name=session["user"],
                             total_users=total_users,
                             active_users=active_users,
                             user_stats=user_stats)
    return redirect("/login")

# Practice Page
@app.route("/practice/<level>")
def practice(level):
    if "user" not in session:
        return redirect("/login")
    
    # Define poses for each level
    poses_config = {
        "easy": ["Mountain Pose", "Tree Pose"],
        "medium": ["Downward Dog", "Triangle Pose"], 
        "difficult": ["Warrior II", "Chair Pose", "Plank"]
    }
    
    if level not in poses_config:
        return redirect("/levels")
    
    poses = poses_config[level]
    return render_template("practice.html", 
                         poses=poses, 
                         level=level, 
                         total_poses=len(poses))

# Levels Page
@app.route("/levels")
def levels():
    if "user" in session:
        return render_template("levels.html")
    return redirect("/login")

# Features Page
@app.route("/features")
def features():
    return render_template("features.html")

# About Page
@app.route("/about")
def about():
    return render_template("about.html")

# Pose detection endpoint
@app.route("/detect_pose", methods=["POST"])
def detect_pose():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    load_models()
    if model is None or encoder is None:
        return jsonify({"error": "Models not loaded"}), 500
    
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data"}), 400
    
    try:
        img = dataurl_to_image(data["image"])
        pose_name, confidence = predict_pose_from_image(img)
        
        return jsonify({
            "pose": pose_name,
            "confidence": confidence,
            "correct": pose_name != "Incorrect Pose" and confidence >= 0.5
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Save session data
@app.route("/save_session", methods=["POST"])
def save_session():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No session data"}), 400
    
    try:
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        
        # Get user ID
        c.execute("SELECT id FROM users WHERE name=?", (session["user"],))
        user_id = c.fetchone()[0]
        
        # Insert session data
        c.execute("""
            INSERT INTO sessions (user_id, accuracy, streak, level)
            VALUES (?, ?, ?, ?)
        """, (user_id, data.get("accuracy", 0), data.get("streak", 0), data.get("level", "easy")))
        
        conn.commit()
        conn.close()
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Logout
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)