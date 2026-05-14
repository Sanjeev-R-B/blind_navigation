import threading
import queue
import subprocess


class TTSEngine:
    def __init__(self, rate=150, volume=1.0):
        self._queue   = queue.Queue()
        self._running = False
        self._thread  = None
        self._rate    = rate

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("TTS engine started.")

    def stop(self):
        self._running = False
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=2)
        print("TTS engine stopped.")

    def speak(self, text, priority=False):
        if priority:
            self._clear_queue()
        self._queue.put(text)

    def _clear_queue(self):
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _worker(self):
        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
                if text is None:
                    break
                # Use Windows SAPI directly via PowerShell — most reliable on Windows
                ps_script = (
                    f"Add-Type -AssemblyName System.Speech; "
                    f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$s.Rate = {int((self._rate - 150) / 10)}; "
                    f"$s.Speak('{text}');"
                )
                subprocess.run(
                    ["powershell", "-Command", ps_script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS error: {e}")
                continue


