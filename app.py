"""Flask app that classifies uploaded images with a Keras model."""

from flask import Flask, render_template, request
from model import preprocess_img, predict_result

app = Flask(__name__)


@app.route("/")
def main():
    """Render the upload page."""
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict_image_file():
    """Classify the uploaded image and render the result page."""
    try:
        img = preprocess_img(request.files["file"].stream)
        pred = predict_result(img)
        return render_template("result.html", predictions=str(pred))
    except (OSError, ValueError, KeyError):
        # OSError covers unreadable or non-image files, KeyError a missing
        # "file" field, ValueError a bad image shape.
        return render_template("result.html", err="File cannot be processed.")


if __name__ == "__main__":
    app.run(port=9000, debug=True)
