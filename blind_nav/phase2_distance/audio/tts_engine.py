import threading
import queue
import subprocess
import time


class TTSEngine:
    def __init__(self, rate=150, volume=1.0):
        self._queue   = queue.Queue()
        self._running = False
        self._thread  = None
        self._rate    = rate

    def start(self):
        ps = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = {int((self._rate - 150) / 10)}; "
            "[Console]::Out.WriteLine('ready'); [Console]::Out.Flush(); "
            "while (($l = [Console]::In.ReadLine()) -ne $null) { "
            "$s.Speak($l); [Console]::Out.WriteLine('done'); [Console]::Out.Flush() }"
        )
        self._proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", ps],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1
        )
        self._proc.stdout.readline()   # wait until PowerShell is ready
        self._running = True
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("TTS engine started.")

    def stop(self):
        self._running = False
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=2)
        if getattr(self, "_proc", None):
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            self._proc.terminate()
        print("TTS engine stopped.")

    def speak(self, text, priority=False):
        if priority:
            # High-priority: flush backlog then jump to front
            self._clear_queue()
            self._queue.put(text)
        elif self._queue.empty():
            # Non-priority: only queue if nothing is already waiting.
            # If the worker is busy, this message is already stale — drop it.
            self._queue.put(text)
        # else: drop silently

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
                print(f"[TTS] {time.strftime('%H:%M:%S')} speaking: {text}")
                self._proc.stdin.write(text.replace("\n", " ") + "\n")
                self._proc.stdin.flush()
                self._proc.stdout.readline()   # blocks until PowerShell reports "done"
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS error: {e}")
                continue