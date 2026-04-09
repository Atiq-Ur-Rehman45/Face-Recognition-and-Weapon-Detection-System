"""
web/routes/stream.py
MJPEG video streaming endpoints.
"""

import time
from flask import Blueprint, Response, current_app
from config import MJPEG_MAX_FPS

stream_bp = Blueprint("stream", __name__)


def _mjpeg_generator(get_frame_fn):
    """Yield MJPEG boundary frames from a callable that returns JPEG bytes."""
    interval = 1.0 / MJPEG_MAX_FPS
    while True:
        frame_bytes = get_frame_fn()
        if frame_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
        time.sleep(interval)


@stream_bp.route("/video_feed")
def video_feed():
    """Main live recognition MJPEG stream consumed by the monitor page."""
    pipeline = current_app.config["VIDEO_PIPELINE"]
    return Response(
        _mjpeg_generator(pipeline.get_frame),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@stream_bp.route("/enrollment_feed")
def enrollment_feed():
    """Enrollment camera MJPEG stream consumed by the enrollment page."""
    svc = current_app.config["ENROLLMENT_SERVICE"]
    return Response(
        _mjpeg_generator(svc.get_frame),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
