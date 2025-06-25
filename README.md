
# Player Re-Identification in a Single Feed 🎥⚽

## 🎯 Objective
Track players in a 15-second football video and maintain consistent IDs using YOLOv11 + DeepSORT.

## 📁 Files
- `best.pt`: Custom YOLOv11 model trained to detect players and the ball.
- `15sec_input_720p.mp4`: Input video clip.
- `output_tracked.mp4`: Output video with persistent player IDs.

## 🚀 Workflow
1. Player detection using YOLOv11.
2. Tracking and Re-ID using DeepSORT.
3. Output video generation with consistent tracking annotations.

## 🛠 Tools Used
- Ultralytics YOLOv11
- DeepSORT Realtime
- OpenCV

## 👨‍💻 Author
Smit Swapnil Patel
"""

# Save to file
with open("README.md", "w") as f:
    f.write(readme_content)
