# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Virtual touchpad position in webcam feed (default position)
TOUCHPAD_TOP_LEFT = (100, 100)
TOUCHPAD_WIDTH = 300
TOUCHPAD_HEIGHT = 200

# Depth threshold in mm (e.g., touching if closer than this)
TOUCH_DEPTH_THRESHOLD = 600

# Smoothing parameters
DEPTH_SMOOTHING_KERNEL = 5
DEPTH_MEDIAN_FILTER_SIZE = 5

# Virtual Screen Calibration (4 corner points, default values)
# Format: (x, y) coordinates
VIRTUAL_SCREEN_CORNERS = [(100, 100), (400, 100), (100, 300), (400, 300)]

# Optional: Depth Threshold for Virtual Space (if needed)
HOVER_DEPTH_THRESHOLD = 800
VIRTUAL_SCREEN_DEPTH_THRESHOLD = 300  # Can be modified during calibration
