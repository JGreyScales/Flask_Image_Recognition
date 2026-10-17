"""Combined pytest suite for the Flask hand-digit image classifier.

Covers the home page, valid uploads, missing files, invalid or unexpected
input, edge cases, and the preprocessing and prediction functions.

Run from the repository root (model.py loads digit_model.h5 by relative path):
    pytest -v
"""

import io

import numpy as np
import pytest
from PIL import Image

from model import preprocess_img, predict_result

ERROR_TEXT = b"File cannot be processed."

# Real sample images shipped in the repository
DIGIT_IMAGES = [
    "test_images/2/Sign 2 (97).jpeg",
    "test_images/4/Sign 4 (92).jpeg",
    "test_images/5/Sign 5 (86).jpeg",
    "test_images/7/Sign 7 (54).jpeg",
]
VALID_IMAGE = DIGIT_IMAGES[0]


def make_image(mode="RGB", size=(64, 64), fmt="PNG"):
    """Build an in-memory image so tests don't depend on files on disk."""
    buf = io.BytesIO()
    Image.new(mode, size).save(buf, format=fmt)
    buf.seek(0)
    return buf


def post_file(client, fileobj, filename, **kwargs):
    """POST a file to the prediction route as multipart form data."""
    return client.post(
        "/prediction",
        data={"file": (fileobj, filename)},
        content_type="multipart/form-data",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Home page
# ---------------------------------------------------------------------------

def test_home_page_loads(client):
    """GET / returns the upload page."""
    response = client.get("/")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# 2. Valid requests
# ---------------------------------------------------------------------------

def test_valid_upload_returns_result_page(client):
    """A real digit image is classified without an error message."""
    with open(VALID_IMAGE, "rb") as img:
        response = post_file(client, img, "sign2.jpeg")
    assert response.status_code == 200
    assert ERROR_TEXT not in response.data


@pytest.mark.parametrize("path", DIGIT_IMAGES)
def test_sequential_valid_uploads(client, path):
    """Each of several real images is processed independently."""
    with open(path, "rb") as img:
        response = post_file(client, img, "digit.jpeg")
    assert response.status_code == 200
    assert ERROR_TEXT not in response.data


def test_upload_with_unexpected_header_still_works(client):
    """Extra request headers don't interfere with a valid upload."""
    with open(VALID_IMAGE, "rb") as img:
        response = post_file(
            client, img, "sign2.jpeg", headers={"X-Unexpected-Header": "value"}
        )
    assert response.status_code == 200
    assert ERROR_TEXT not in response.data


# ---------------------------------------------------------------------------
# 3. Missing file
# ---------------------------------------------------------------------------

def test_missing_file_is_handled(client):
    """POST with no 'file' field shows the error message instead of crashing."""
    response = client.post(
        "/prediction", data={}, content_type="multipart/form-data"
    )
    assert response.status_code == 200
    assert ERROR_TEXT in response.data


# ---------------------------------------------------------------------------
# 4. Invalid or unexpected input
# ---------------------------------------------------------------------------

def test_non_image_upload_is_handled(client):
    """A text file uploaded as an image is rejected gracefully."""
    response = post_file(
        client, io.BytesIO(b"this is not an image"), "invalid_file.txt"
    )
    assert response.status_code == 200
    assert ERROR_TEXT in response.data


def test_fake_image_bytes_are_rejected(client):
    """Garbage bytes with a .jpg name are rejected, not reported as a result."""
    response = post_file(client, io.BytesIO(b"fake_image_data"), "test.jpg")
    assert response.status_code == 200
    assert ERROR_TEXT in response.data


def test_empty_file_is_handled(client):
    """A zero-byte upload is rejected gracefully."""
    response = post_file(client, io.BytesIO(b""), "empty.jpg")
    assert response.status_code == 200
    assert ERROR_TEXT in response.data


def test_wrong_channel_count_is_handled(client):
    """An RGBA image has 4 channels, so reshaping fails; the route reports it."""
    response = post_file(client, make_image(mode="RGBA"), "alpha.png")
    assert response.status_code == 200
    assert ERROR_TEXT in response.data


def test_get_on_prediction_route_not_allowed(client):
    """The prediction route only accepts POST."""
    assert client.get("/prediction").status_code == 405


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_large_valid_image_is_processed(client):
    """A very large but valid image is resized and classified normally."""
    response = post_file(
        client, make_image(size=(4000, 4000), fmt="JPEG"), "large.jpg"
    )
    assert response.status_code == 200
    assert ERROR_TEXT not in response.data


def test_non_standard_extension_with_valid_content(client):
    """The file is judged by its content, not its extension."""
    response = post_file(client, make_image(fmt="PNG"), "picture.dat")
    assert response.status_code == 200
    assert ERROR_TEXT not in response.data


# ---------------------------------------------------------------------------
# 5. Preprocessing and prediction
# ---------------------------------------------------------------------------

def test_preprocess_real_image_shape_and_range():
    """A real sample image becomes a normalized (1, 224, 224, 3) batch."""
    result = preprocess_img(VALID_IMAGE)
    assert result.shape == (1, 224, 224, 3)
    assert np.min(result) >= 0.0
    assert np.max(result) <= 1.0


def test_preprocess_in_memory_image_shape_and_range():
    """Images of any size are resized to 224x224 and scaled to 0-1."""
    result = preprocess_img(make_image(size=(100, 80)))
    assert result.shape == (1, 224, 224, 3)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocess_invalid_path_raises():
    """A path that doesn't exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        preprocess_img("invalid/path/to/image.jpeg")


def test_preprocess_rejects_non_image():
    """Bytes that aren't an image raise an OSError (PIL.UnidentifiedImageError)."""
    with pytest.raises(OSError):
        preprocess_img(io.BytesIO(b"not an image"))


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_preprocess_rejects_non_rgb_images(mode):
    """Grayscale and RGBA images don't have 3 channels, so reshaping fails."""
    with pytest.raises(ValueError):
        preprocess_img(make_image(mode=mode))


@pytest.mark.parametrize("path", DIGIT_IMAGES)
def test_predict_result_returns_class_index(path):
    """Prediction is a non-negative integer class index."""
    prediction = predict_result(preprocess_img(path))
    assert isinstance(prediction, (int, np.integer))
    assert prediction >= 0


def test_predictions_are_consistent():
    """The same input always produces the same prediction."""
    processed = preprocess_img(DIGIT_IMAGES[-1])
    predictions = [predict_result(processed) for _ in range(5)]
    assert all(p == predictions[0] for p in predictions)