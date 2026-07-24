"""
calibrate_lawn_map.py

Interactive lawn-plane calibration for a fixed, elevated OAK-D Lite.

This version works when launched using the green Run button in Thonny.

Workflow:
    1. Run this file.
    2. Enter the real lawn width in metres.
    3. Enter the real lawn length in metres.
    4. The live OAK-D camera window opens.
    5. Click the four lawn corners in this exact order:

           1. FAR LEFT
           2. FAR RIGHT
           3. NEAR RIGHT
           4. NEAR LEFT

       In the displayed image, this will normally be:

           1. Top-left
           2. Top-right
           3. Bottom-right
           4. Bottom-left

    6. Press S to save.

Controls:
    Left-click = add corner
    R          = reset all selected corners
    U          = undo the last corner
    S          = save calibration
    Q          = quit without saving

Output:
    lawn_calibration.json
"""

import json
from pathlib import Path

import cv2
import numpy as np

from trim_detector import build_pipeline
from camera_session import CameraSession


# ---------------------------------------------------------------------------
# Paths and display settings
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent

CALIBRATION_FILE = HERE / "lawn_calibration.json"

WINDOW_NAME = "Lawn calibration - select FAR-L, FAR-R, NEAR-R, NEAR-L"

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Measurement input
# ---------------------------------------------------------------------------

def request_positive_measurement(prompt):
    """
    Ask the user for a positive measurement in metres.

    Continues asking until a valid positive number is entered.
    """

    while True:

        raw_value = input(prompt).strip()

        try:
            value = float(raw_value)

        except ValueError:
            print(
                "Please enter a number in metres, "
                "for example: 4.2"
            )
            continue

        if not np.isfinite(value):
            print("The value must be a finite number.")
            continue

        if value <= 0:
            print("The value must be greater than zero.")
            continue

        return value


def request_lawn_measurements():
    """
    Request the real lawn dimensions from the Thonny Shell.
    """

    print()
    print("=" * 64)
    print("FIXED OAK-D LAWN MAP CALIBRATION")
    print("=" * 64)
    print()
    print(
        "Measure the rectangular lawn area visible to the camera."
    )
    print()
    print(
        "WIDTH means the real distance from the LEFT lawn edge "
        "to the RIGHT lawn edge."
    )
    print()
    print(
        "LENGTH means the real distance from the NEAR lawn edge "
        "to the FAR lawn edge."
    )
    print()

    width_m = request_positive_measurement(
        "Enter the lawn WIDTH in metres: "
    )

    length_m = request_positive_measurement(
        "Enter the lawn LENGTH in metres: "
    )

    print()
    print("Measurements entered:")
    print(f"  Width:  {width_m:.3f} metres")
    print(f"  Length: {length_m:.3f} metres")
    print()

    return width_m, length_m


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def polygon_area(points):
    """
    Return the absolute image-space area of a polygon.

    This is used only to reject a calibration where all selected coordinates
    are effectively on the same line or occupy almost no image area.
    """

    array = np.asarray(points, dtype=np.float32)

    if len(array) < 3:
        return 0.0

    return abs(float(cv2.contourArea(array)))


def build_homography(image_points, lawn_width_m, lawn_length_m):
    """
    Build the transformation from image pixels to lawn coordinates.

    Selected image point order:
        1 far-left
        2 far-right
        3 near-right
        4 near-left

    Lawn coordinate system:
        X = metres right from the lawn's left edge
        Y = metres away from the lawn's near edge

    Therefore:
        near-left  = (0, 0)
        near-right = (width, 0)
        far-right  = (width, length)
        far-left   = (0, length)
    """

    source_points = np.asarray(
        image_points,
        dtype=np.float32,
    )

    destination_points = np.asarray(
        [
            [0.0, lawn_length_m],
            [lawn_width_m, lawn_length_m],
            [lawn_width_m, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    homography = cv2.getPerspectiveTransform(
        source_points,
        destination_points,
    )

    return homography


def verify_homography(
    homography,
    image_points,
    lawn_width_m,
    lawn_length_m,
):
    """
    Verify that the selected image corners transform to the expected lawn
    coordinates within a small numerical tolerance.
    """

    source = np.asarray(
        [image_points],
        dtype=np.float32,
    )

    projected = cv2.perspectiveTransform(
        source,
        homography,
    )[0]

    expected = np.asarray(
        [
            [0.0, lawn_length_m],
            [lawn_width_m, lawn_length_m],
            [lawn_width_m, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    errors = np.linalg.norm(
        projected - expected,
        axis=1,
    )

    maximum_error = float(np.max(errors))

    return maximum_error


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def draw_multiline_text(
    image,
    lines,
    start_x,
    start_y,
    colour,
    scale=0.48,
    thickness=1,
    spacing=21,
):
    """
    Draw several lines of text on an OpenCV image.
    """

    y = start_y

    for line in lines:

        cv2.putText(
            image,
            line,
            (start_x, y),
            FONT,
            scale,
            colour,
            thickness,
            cv2.LINE_AA,
        )

        y += spacing


def draw_calibration_display(
    frame,
    points,
    lawn_width_m,
    lawn_length_m,
    status_message,
):
    """
    Draw selected points and instructions over the live camera feed.
    """

    display = frame.copy()

    image_height, image_width = display.shape[:2]

    point_names = [
        "1 FAR-LEFT",
        "2 FAR-RIGHT",
        "3 NEAR-RIGHT",
        "4 NEAR-LEFT",
    ]

    point_colours = [
        (0, 255, 255),
        (0, 220, 255),
        (0, 165, 255),
        (80, 255, 80),
    ]

    # Draw the selected corners.
    for index, point in enumerate(points):

        x = int(round(point[0]))
        y = int(round(point[1]))

        colour = point_colours[index]

        cv2.circle(
            display,
            (x, y),
            8,
            colour,
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            display,
            (x, y),
            13,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        label_x = min(
            max(8, x + 12),
            max(8, image_width - 170),
        )

        label_y = min(
            max(20, y - 10),
            max(20, image_height - 10),
        )

        cv2.putText(
            display,
            point_names[index],
            (label_x, label_y),
            FONT,
            0.52,
            colour,
            2,
            cv2.LINE_AA,
        )

    # Draw lines joining the selected points.
    if len(points) >= 2:

        selected_array = np.asarray(
            points,
            dtype=np.int32,
        )

        cv2.polylines(
            display,
            [selected_array],
            len(points) == 4,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Draw a translucent instruction panel.
    overlay = display.copy()

    panel_height = 116

    cv2.rectangle(
        overlay,
        (0, 0),
        (image_width, panel_height),
        (15, 15, 15),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.82,
        display,
        0.18,
        0,
        display,
    )

    next_instruction = (
        point_names[len(points)]
        if len(points) < 4
        else "All corners selected. Press S to save."
    )

    instruction_lines = [
        "Click in order: FAR-LEFT, FAR-RIGHT, NEAR-RIGHT, NEAR-LEFT",
        f"Next: {next_instruction}",
        (
            f"Selected: {len(points)}/4   "
            f"Lawn: {lawn_width_m:.3f} m x "
            f"{lawn_length_m:.3f} m"
        ),
        "Keys: R reset   U undo   S save   Q quit",
        status_message,
    ]

    draw_multiline_text(
        display,
        instruction_lines,
        8,
        20,
        (0, 255, 255),
        scale=0.45,
        thickness=1,
        spacing=21,
    )

    return display


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_calibration(
    frame,
    points,
    lawn_width_m,
    lawn_length_m,
):
    """
    Validate and save the complete calibration.
    """

    if len(points) != 4:
        print(
            "Calibration cannot be saved until exactly four "
            "corners have been selected."
        )
        return False

    selected_area_px = polygon_area(points)

    if selected_area_px < 100.0:
        print(
            "The selected lawn polygon is too small or invalid. "
            "Press R and select the four corners again."
        )
        return False

    homography = build_homography(
        points,
        lawn_width_m,
        lawn_length_m,
    )

    if not np.all(np.isfinite(homography)):
        print(
            "The transformed calibration contains invalid values. "
            "Press R and select the corners again."
        )
        return False

    verification_error = verify_homography(
        homography,
        points,
        lawn_width_m,
        lawn_length_m,
    )

    calibration = {
        "version": 2,
        "coordinate_system": {
            "units": "metres",
            "x_axis": (
                "positive right from the lawn's left edge"
            ),
            "y_axis": (
                "positive away from the lawn's near edge"
            ),
            "origin": "near-left lawn corner",
        },
        "lawn_width_m": float(lawn_width_m),
        "lawn_length_m": float(lawn_length_m),
        "image_points_order": [
            "far-left",
            "far-right",
            "near-right",
            "near-left",
        ],
        "image_points_far_left_far_right_near_right_near_left": [
            [float(x), float(y)]
            for x, y in points
        ],
        # Included for compatibility with the original map scripts.
        "image_points_tl_tr_br_bl": [
            [float(x), float(y)]
            for x, y in points
        ],
        "homography_image_to_lawn": (
            homography.astype(float).tolist()
        ),
        "camera_width_px": int(frame.shape[1]),
        "camera_height_px": int(frame.shape[0]),
        "selected_polygon_area_px": selected_area_px,
        "corner_verification_max_error_m": (
            verification_error
        ),
    }

    temporary_file = CALIBRATION_FILE.with_suffix(
        ".json.tmp"
    )

    temporary_file.write_text(
        json.dumps(
            calibration,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_file.replace(CALIBRATION_FILE)

    confirmation_image = frame.copy()

    polygon = np.asarray(
        points,
        dtype=np.int32,
    )

    cv2.polylines(
        confirmation_image,
        [polygon],
        True,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )

    corner_names = [
        "FAR-LEFT",
        "FAR-RIGHT",
        "NEAR-RIGHT",
        "NEAR-LEFT",
    ]

    for name, point in zip(
        corner_names,
        points,
    ):

        x = int(round(point[0]))
        y = int(round(point[1]))

        cv2.circle(
            confirmation_image,
            (x, y),
            8,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            confirmation_image,
            name,
            (x + 10, max(18, y - 10)),
            FONT,
            0.48,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    confirmation_path = (
        HERE / "lawn_calibration_preview.jpg"
    )

    cv2.imwrite(
        str(confirmation_path),
        confirmation_image,
    )

    print()
    print("=" * 64)
    print("CALIBRATION SAVED")
    print("=" * 64)
    print(f"Calibration file: {CALIBRATION_FILE}")
    print(f"Preview image:    {confirmation_path}")
    print(f"Lawn width:       {lawn_width_m:.3f} m")
    print(f"Lawn length:      {lawn_length_m:.3f} m")
    print(
        "Corner verification maximum error: "
        f"{verification_error:.8f} m"
    )
    print()
    print("The camera must now remain in exactly this position.")
    print()

    return True


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main():
    """
    Run interactive OAK-D lawn calibration.
    """

    lawn_width_m, lawn_length_m = (
        request_lawn_measurements()
    )

    print("Opening OAK-D Lite camera...")
    print()

    try:
        camera = CameraSession(build_pipeline)

    except Exception as error:
        print(
            "The OAK-D camera could not be started:"
        )
        print(error)
        return

    selected_points = []

    latest_frame = None

    status_message = (
        "Waiting for the first corner: FAR-LEFT"
    )

    def mouse_callback(
        event,
        x,
        y,
        flags,
        parameter,
    ):
        """
        Record one corner for each left mouse click.
        """

        nonlocal status_message

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(selected_points) >= 4:
            status_message = (
                "Four corners already selected. "
                "Press S to save or R to reset."
            )
            return

        selected_points.append(
            [float(x), float(y)]
        )

        corner_names = [
            "FAR-LEFT",
            "FAR-RIGHT",
            "NEAR-RIGHT",
            "NEAR-LEFT",
        ]

        selected_name = corner_names[
            len(selected_points) - 1
        ]

        print(
            f"Selected {selected_name}: "
            f"pixel ({x}, {y})"
        )

        if len(selected_points) < 4:
            next_name = corner_names[
                len(selected_points)
            ]

            status_message = (
                f"Selected {selected_name}. "
                f"Next click: {next_name}"
            )

        else:
            status_message = (
                "All four corners selected. "
                "Press S to save."
            )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback,
    )

    try:

        while True:

            frame_group = camera.get()

            if frame_group is not None:

                latest_frame = (
                    frame_group["rgb"].getCvFrame()
                )

            elif camera.crashed():

                print(
                    "[camera] stopped delivering frames; "
                    "attempting restart"
                )

                if not camera.restart():
                    print(
                        "[camera] restart failed"
                    )
                    break

                status_message = (
                    "Camera restarted. Calibration points "
                    "were retained."
                )

            if latest_frame is None:

                key = cv2.waitKey(1) & 255

                if key == ord("q"):
                    break

                continue

            display = draw_calibration_display(
                latest_frame,
                selected_points,
                lawn_width_m,
                lawn_length_m,
                status_message,
            )

            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            key = cv2.waitKey(1) & 255

            if key == ord("q"):

                print(
                    "Calibration cancelled without saving."
                )

                break

            if key == ord("r"):

                selected_points.clear()

                status_message = (
                    "Points reset. Click FAR-LEFT first."
                )

                print(
                    "All selected corners were reset."
                )

            elif key == ord("u"):

                if selected_points:

                    removed = selected_points.pop()

                    print(
                        "Removed the last corner:",
                        removed,
                    )

                    corner_names = [
                        "FAR-LEFT",
                        "FAR-RIGHT",
                        "NEAR-RIGHT",
                        "NEAR-LEFT",
                    ]

                    status_message = (
                        "Last point removed. "
                        f"Next click: "
                        f"{corner_names[len(selected_points)]}"
                    )

                else:

                    status_message = (
                        "There is no selected point to undo."
                    )

            elif key == ord("s"):

                saved = save_calibration(
                    latest_frame,
                    selected_points,
                    lawn_width_m,
                    lawn_length_m,
                )

                if saved:
                    break

                status_message = (
                    "Calibration was not saved. "
                    "Check the Shell message."
                )

    finally:

        camera.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()