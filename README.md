# DeepFace 情绪识别 快速示例

说明
- 使用 DeepFace 库做情绪识别（愤怒、喜悦、悲伤、惊讶、恐惧、厌恶、平静）
- 提供两个脚本：
  - scripts/deepface_webcam.py：实时摄像头情绪识别（带画面叠加）
  - scripts/deepface_image.py：分析单张图片并打印情绪分布
- DeepFace 会自动下载所需模型（第一次运行会从网络下载，需联网）

环境准备（推荐使用虚拟环境）
- Linux / macOS:
  1. python3 -m venv venv
  2. source venv/bin/activate
  3. pip install -r requirements.txt
- Windows (PowerShell):
  1. python -m venv venv
  2. .\venv\Scripts\Activate.ps1
  3. pip install -r requirements.txt

requirements.txt 包含 DeepFace 和 OpenCV。若你有 GPU 并想用 TensorFlow GPU，请自行安装合适版本的 `tensorflow`（例如 `pip install tensorflow` 或 `pip install tensorflow-gpu`，注意兼容性）。

运行示例

1) 实时摄像头（推荐）
- 启动：
  - 在项目根目录运行：
    python scripts/deepface_webcam.py
- 按键：
  - 按 `q` 退出

2) 单张图片
- 用法：
  - python scripts/deepface_image.py --img path/to/photo.jpg
- 输出示例：
  - Dominant emotion: happy
  - emotions: {'angry': 0.001, 'disgust': 0.000, 'fear': 0.002, 'happy': 0.95, 'sad': 0.02, 'surprise': 0.01, 'neutral': 0.017}

常见问题
- 摄像头打不开：确认是否被其他程序占用，或尝试更改 `camera_index`（脚本顶部变量）。
- DeepFace 下载模型慢：首次运行需要下载模型文件（几十 MB），请确保网络通畅。
- 性能：实时推理速度取决于机器和模型，可通过降低处理频率（脚本中有 frame_skip 配置）改善。
- 权限：在 macOS 上需要允许终端或 Python 应用访问摄像头。

如果你想我把这些文件直接推到仓库（s0531152391-byte/exmp），并且是否要放在根目录或 `scripts/` 子目录，请回复确认.
