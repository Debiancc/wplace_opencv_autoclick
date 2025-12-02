"""
YOLO Vision Project with Screenshot Hotkey Support
Captures screenshots using Ctrl+Shift+Q hotkey and optionally performs object detection.
"""
import argparse
from screenshot_capture import ScreenshotCapture
from hotkey_listener import HotkeyListener
from square_detector import SquareDetector
from pynput.mouse import Button, Controller
import time

# Optional YOLO import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
"""
YOLO Vision Project with Screenshot Hotkey Support
Captures screenshots using Ctrl+Shift+Q hotkey and optionally performs object detection.
"""
import argparse
from screenshot_capture import ScreenshotCapture
from hotkey_listener import HotkeyListener
from square_detector import SquareDetector
from pynput.mouse import Button, Controller
import time

# Optional YOLO import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("警告: ultralytics 未安装，YOLO 检测功能不可用")


class YOLOScreenshotApp:
    """Main application integrating screenshot capture and YOLO detection."""
    
    def __init__(self, enable_detection=False, enable_square_detection=False, click_delay=0.5):
        """
        Initialize the application.
        
        Args:
            enable_detection (bool): Whether to perform YOLO detection on screenshots
            enable_square_detection (bool): Whether to detect square points
            click_delay (float): Delay in seconds between clicks
        """
        self.enable_detection = enable_detection
        self.enable_square_detection = enable_square_detection
        self.click_delay = click_delay
        self.screenshot_capture = ScreenshotCapture()
        
        # Cleanup screenshots directory on startup
        self.screenshot_capture.clear_directory()
        
        self.model = None
        self.square_detector = None
        self.mouse = Controller()
        
        if enable_detection:
            if not YOLO_AVAILABLE:
                print("错误: YOLO 检测已启用但 ultralytics 未安装")
                print("请运行: pip install ultralytics")
                self.enable_detection = False
            else:
                print("加载 YOLO 模型...")
                self.model = YOLO('yolov8n.pt')
                print("YOLO 模型加载成功！")
        
        if enable_square_detection:
            print("初始化方块点检测器...")
            self.square_detector = SquareDetector()
            print("方块点检测器初始化成功！")
    
    def on_screenshot_hotkey(self):
        """Callback function triggered when screenshot hotkey is pressed."""
        try:
            print("\n" + "="*50)
            print("捕获截图中...")
            
            # Capture and save screenshot (now returns filepath and region)
            filepath, region = self.screenshot_capture.capture_and_save()
            
            if filepath is None:
                print("截图已取消")
                print("="*50 + "\n")
                return
            
            # Perform square point detection if enabled
            if self.enable_square_detection and self.square_detector:
                print("\n检测方块点中...")
                from PIL import Image
                img = Image.open(filepath)
                print(f"✓ 已加载图像: {img.size}")
                
                # Detect squares
                detections = self.square_detector.detect_squares(img)
                
                if detections:
                    # Calculate absolute coordinates
                    x1, y1, x2, y2 = region
                    detections = self.square_detector.calculate_absolute_coordinates(
                        detections, 
                        region_offset=(x1, y1)
                    )
                    print(f"✓ 已计算绝对坐标")
                    
                    # --- Mouse Interaction: Click all squares ---
                    print(f"\n✓ 准备点击 {len(detections)} 个方块 (延迟: {self.click_delay}s)...")
                    print("  提示: 移动鼠标可中断点击")
                    
                    for i, det in enumerate(detections):
                        target_x, target_y = det['absolute_coords']
                        print(f"  [{i+1}/{len(detections)}] 点击方块: ({target_x}, {target_y})")
                        
                        # Move mouse
                        self.mouse.position = (target_x, target_y)
                        
                        # Safety check: Wait and check if mouse moved
                        check_interval = 0.05
                        elapsed = 0
                        interrupted = False
                        
                        while elapsed < self.click_delay:
                            time.sleep(check_interval)
                            elapsed += check_interval
                            
                            # Check if mouse moved significantly from target
                            curr_x, curr_y = self.mouse.position
                            if abs(curr_x - target_x) > 5 or abs(curr_y - target_y) > 5:
                                print("\n!!! 检测到鼠标移动，中断操作 !!!")
                                interrupted = True
                                break
                        
                        if interrupted:
                            break
                            
                        # Perform click
                        self.mouse.click(Button.left)
                    
                    if not interrupted:
                        print(f"✓ 所有方块点击完成")
                    
                    # Visualize
                    print(f"✓ 开始生成可视化...")
                    visualized = self.square_detector.visualize_detections(img, detections)
                    vis_path = filepath.replace('.png', '_squares.png')
                    visualized.save(vis_path)
                    print(f"✓ 可视化结果已保存: {vis_path}")
                    
                    # Print summary
                    print(f"\n方块点详细信息 (前10个):")
                    for i, det in enumerate(detections[:10]):
                        abs_x, abs_y = det['absolute_coords']
                        brightness = det.get('brightness', 'N/A')
                        filled = det.get('filled_ratio', 'N/A')
                        print(f"  点 {i+1}: 屏幕坐标 ({abs_x}, {abs_y}) - 面积: {det['area']}px - 亮度: {brightness:.1f} - 填充率: {filled:.2f}")
                    
                    if len(detections) > 10:
                        print(f"  ... 还有 {len(detections) - 10} 个方块点")
                    
                    print(f"\n✓ 总计: {len(detections)} 个方块点")
                else:
                    print("未检测到方块点")
            
            # Optionally perform YOLO detection
            if self.enable_detection and self.model:
                print("\n运行 YOLO 目标检测...")
                results = self.model(filepath)
                
                # Save detection results
                for result in results:
                    result_path = filepath.replace('.png', '_yolo.jpg')
                    result.save(filename=result_path)
                    print(f"YOLO 检测结果已保存: {result_path}")
                    
                    # Print detected objects
                    if result.boxes:
                        print(f"检测到 {len(result.boxes)} 个对象:")
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            cls_name = self.model.names[cls_id]
                            print(f"  - {cls_name}: {conf:.2f}")
                    else:
                        print("未检测到对象")
            
            print("\n截图完成！")
            print("="*50 + "\n")
            
        except Exception as e:
            import traceback
            print(f"捕获截图时出错: {e}")
            traceback.print_exc()
            print("="*50 + "\n")
    
    def run(self):
        """Start the application and listen for hotkeys."""
        print("\n" + "="*60)
        print("YOLO 截图应用程序")
        print("="*60)
        print(f"YOLO 检测: {'启用' if self.enable_detection else '禁用'}")
        print(f"方块点检测: {'启用' if self.enable_square_detection else '禁用'}")
        print(f"点击延迟: {self.click_delay}秒")
        print("="*60 + "\n")
        
        # Create and start hotkey listener
        listener = HotkeyListener(hotkey_callback=self.on_screenshot_hotkey)
        listener.start()


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='截图工具支持可选的 YOLO 目标检测和方块点检测'
    )
    parser.add_argument(
        '--detect',
        action='store_true',
        help='启用 YOLO 目标检测'
    )
    parser.add_argument(
        '--detect-squares',
        action='store_true',
        help='启用方块点检测和计数'
    )
    parser.add_argument(
        '--click-delay',
        type=float,
        default=0.5,
        help='自动点击的间隔延迟(秒)，默认 0.5'
    )
    
    args = parser.parse_args()
    
    # Create and run application
    app = YOLOScreenshotApp(
        enable_detection=args.detect,
        enable_square_detection=True,
        click_delay=0.1
    )
    app.run()


if __name__ == '__main__':
    main()
