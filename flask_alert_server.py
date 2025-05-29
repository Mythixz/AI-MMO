from flask import Flask, render_template, Response
import threading
import time
import queue

app = Flask(__name__)
q = queue.Queue()

@app.route("/")
def index():
    return render_template("index.html") 

@app.route("/stream")
def stream():
    def event_stream():
        while True:
            msg = q.get()
            yield f"data: {msg}\n\n"
    return Response(event_stream(), mimetype="text/event-stream")

def run_flask():
    app.run(debug=False, port=5000, use_reloader=False)

if __name__ == "__main__":
    run_flask()
