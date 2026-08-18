# tests/manual/test_speech.py
"""手动测试 SpeechProvider - 实时语音识别"""

import time
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.nan_itself.providers.listen import SpeechProvider


def test_speech_realtime():
    """
    测试实时语音识别（需要麦克风）
    
    对着麦克风说话，程序会实时打印识别结果。
    按 Ctrl+C 退出。
    """
    
    print("🎤 初始化 SpeechProvider...")
    
    try:
        speech = SpeechProvider(
            models_dir=Path("./models/whispercpp"),
            model_size="base",  # 用 base 模型，速度快，适合实时
            queue_maxsize=20,
            n_threads=4,
        )
        print("✅ SpeechProvider 初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎙️  开始语音识别（实时模式）")
    print("   请对着麦克风说话...")
    print("   每句话结束后会自动识别并打印")
    print("   按 Ctrl+C 停止")
    print("=" * 50 + "\n")
    
    # 启动识别
    speech.start()
    print("✅ 语音识别已启动\n")
    
    try:
        # 持续轮询，打印识别结果
        while True:
            messages = speech.get_messages()
            if messages:
                for msg in messages:
                    print(f"📝 [{time.strftime('%H:%M:%S')}] {msg}")
            time.sleep(0.1)  # 避免 CPU 空转
            
    except KeyboardInterrupt:
        print("\n\n⏹️  收到停止信号，正在关闭...")
    finally:
        speech.close()
        print("✅ 语音识别已停止")
    
    return True


def test_speech_with_audio_file():
    """
    测试语音识别 - 处理已有的音频文件（离线模式）
    
    注意：pywhispercpp 的 Assistant 主要用于实时流，
    处理文件可以用 WhisperASR（离线版）。
    这里演示如果要用 Assistant 处理文件，需要改造。
    """
    print("📁 SpeechProvider 主要面向实时语音流。")
    print("   如需处理音频文件，建议使用 WhisperASR（离线版）。")
    print("   或者用 pywhispercpp 的 Model 直接 transcribe。")
    
    # 简单的文件转写示例（用底层 Model，不经过 Assistant）
    try:
        from pywhispercpp.model import Model
        
        audio_path = Path("test_audio.wav")
        if not audio_path.exists():
            print(f"⚠️  测试音频不存在: {audio_path}")
            print("   跳过文件转写测试")
            return
        
        print(f"\n📁 转写音频文件: {audio_path}")
        model = Model("base", models_dir="./models/whispercpp")
        segments = model.transcribe(str(audio_path))
        
        text = " ".join(s.text for s in segments)
        print(f"📝 转写结果: {text}")
        
    except Exception as e:
        print(f"⚠️  文件转写测试失败: {e}")


if __name__ == "__main__":
    print("选择测试模式:")
    print("  1. 实时语音识别（需要麦克风）")
    print("  2. 查看文件转写说明")
    
    choice = input("\n请选择 (1/2，默认1): ").strip() or "1"
    
    if choice == "1":
        test_speech_realtime()
    else:
        test_speech_with_audio_file()