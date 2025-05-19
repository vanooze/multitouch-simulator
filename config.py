# =======================
# Camera Configuration
# =======================
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# =======================
# Touchpad Settings (Legacy/Optional)
# =======================
TOUCHPAD_TOP_LEFT = (100, 100)
TOUCHPAD_WIDTH = 300
TOUCHPAD_HEIGHT = 200

# =======================
# Depth Thresholds (in mm)
# =======================
TOUCH_DEPTH_THRESHOLD = 700         # Depth difference to trigger a touch
HOVER_DEPTH_THRESHOLD = 1500       # Above this, considered hover
VIRTUAL_SCREEN_DEPTH_THRESHOLD = 300

# =======================
# Smoothing Parameters
# =======================
DEPTH_SMOOTHING_KERNEL = 5
DEPTH_MEDIAN_FILTER_SIZE = 5
MOUSE_SMOOTH_FACTOR = 0.2 # 0.0 no smoothing, 1.0 = full smoothing (no movement)

# =======================
# Default Virtual Screen Calibration
# Format: (x, y) coordinates
# =======================
VIRTUAL_SCREEN_CORNERS = [
    (100, 100), (400, 100),
    (100, 300), (400, 300)
]