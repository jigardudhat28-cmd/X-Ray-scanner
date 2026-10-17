"""AI X-Ray Scanner System - Flask web application.
Run with:  python app.py   (opens in Chrome automatically)
"""
import os
import re
import sqlite3
import threading
import uuid
import webbrowser
from datetime import datetime
from functools import wraps

from flask import (Flask, flash, g, jsonify, redirect, render_template, request,
                   send_from_directory, session, url_for)
from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash

from model.predict import ModelNotTrainedError, predict_xray, validate_xray

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "users.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "webp"}
INVALID_MSG = "This image is invalid. Please use a chest X-ray image."
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key-for-your-project")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit


# ---------------------------------------------------------------- database
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            """CREATE TABLE IF NOT EXISTS users (
                   id            INTEGER PRIMARY KEY AUTOINCREMENT,
                   name          TEXT NOT NULL,
                   email         TEXT NOT NULL UNIQUE,
                   password_hash TEXT NOT NULL,
                   created_at    TEXT NOT NULL)"""
        )


# ------------------------------------------------------------------- auth
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify(ok=False, error="Session expired. Please log in again."), 401
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


@app.after_request
def no_cache(resp):
    # stops the browser Back button from showing protected pages after logout
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/")
def index():
    return redirect(url_for("home" if "user_id" in session else "login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("home"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = get_db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"], session["user_name"] = user["id"], user["name"]
            return redirect(url_for("home"))
        flash("Incorrect email/user ID or password. Please try again.", "error")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        pw, confirm = request.form.get("password", ""), request.form.get("confirm", "")
        error = None
        if not name or not email or not pw:
            error = "Please fill in all fields."
        elif not EMAIL_RE.match(email):
            error = "Enter a valid email address."
        elif len(pw) < 6:
            error = "Password must be at least 6 characters."
        elif pw != confirm:
            error = "Passwords do not match."
        if error is None:
            try:
                with get_db() as db:
                    db.execute(
                        "INSERT INTO users (name, email, password_hash, created_at) VALUES (?,?,?,?)",
                        (name, email, generate_password_hash(pw),
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    )
            except sqlite3.IntegrityError:
                error = "An account with this email already exists."
        if error:
            flash(error, "error")
            return render_template("register.html", name=name, email=email)
        flash("Account created. Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ------------------------------------------------------------------ pages
@app.route("/home")
@login_required
def home():
    return render_template("home.html")


@app.route("/about")
@login_required
def about():
    return render_template("about.html")


@app.route("/contact")
@login_required
def contact():
    return render_template("contact.html")


@app.route("/upload")
@login_required
def upload():
    session.pop("scan", None)
    return render_template("upload.html")


@app.route("/uploads/<name>")
@login_required
def uploaded_file(name):
    return send_from_directory(UPLOAD_DIR, name)


@app.route("/api/scan", methods=["POST"])
@login_required
def api_scan():
    """Validate the image, run the CNN, keep the result for the result page."""
    file = request.files.get("xray")
    if not file or not file.filename:
        return jsonify(ok=False, error="Please choose an image first."), 400
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXT:
        return jsonify(ok=False, error=INVALID_MSG)

    name = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(UPLOAD_DIR, name)
    file.save(path)
    try:
        Image.open(path).verify()  # is it really an image?
    except Exception:
        os.remove(path)
        return jsonify(ok=False, error=INVALID_MSG)

    if not validate_xray(path):
        os.remove(path)
        return jsonify(ok=False, error=INVALID_MSG)

    try:
        result = predict_xray(path)
    except ModelNotTrainedError as exc:
        os.remove(path)
        return jsonify(ok=False, error=str(exc)), 500

    session["scan"] = {"file": name, **result}
    return jsonify(ok=True, redirect=url_for("scanning"))


@app.route("/scanning")
@login_required
def scanning():
    if "scan" not in session:
        return redirect(url_for("upload"))
    return render_template("scanning.html", scan=session["scan"])


@app.route("/result")
@login_required
def result():
    if "scan" not in session:
        return redirect(url_for("upload"))
    return render_template("result.html", scan=session["scan"])


@app.errorhandler(413)
def too_large(_e):
    return jsonify(ok=False, error="File is too large (max 10 MB)."), 413


# ------------------------------------------------------------------- main
def open_browser():
    url = "http://127.0.0.1:5000"
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        "/usr/bin/google-chrome", "/usr/bin/chromium-browser",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for p in chrome_paths:
        if os.path.exists(p):
            webbrowser.register("chrome-local", None, webbrowser.BackgroundBrowser(p))
            webbrowser.get("chrome-local").open(url)
            return
    webbrowser.open(url)  # fall back to the default browser


if __name__ == "__main__":
    init_db()
    threading.Timer(1.5, open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
