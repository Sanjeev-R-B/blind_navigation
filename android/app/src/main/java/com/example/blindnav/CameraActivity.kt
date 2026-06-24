package com.example.blindnav

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.util.SizeF
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraActivity : ComponentActivity(), TextToSpeech.OnInitListener {

    private val TAG = "CameraActivity"
    private val CAMERA_PERMISSION_REQUEST_CODE = 1001
    private val INPUT_SIZE = 192

    // Set this to a value > 0 (e.g., 1400f) to manually override the focal length.
    // Set to null to use the mathematically correct automatic calculation.
    private val HARDCODED_FOCAL_LENGTH_PX: Float? = 1400f


    // UI
    private lateinit var viewFinder: PreviewView
    private lateinit var overlayView: OverlayView

    // Camera / inference
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var ortEnv: OrtEnvironment
    private lateinit var ortSession: OrtSession
    private lateinit var ortInputName: String

    // Python postprocessing
    private lateinit var pyModule: PyObject
    private var focalLengthSet = false
    private var executionProvider = "CPU"

    // Android TTS
    private var tts: TextToSpeech? = null
    private var ttsReady = false

    // ── FPS cap: process at most 12 frames per second ────────────────────────
    private val TARGET_FPS           = 12
    private val FRAME_INTERVAL_MS    = 1000L / TARGET_FPS   // 83 ms
    private var lastFrameTimestamp   = 0L

    // Alert rate-limiting: speak every N frames (36 frames × 83ms ≈ 3 seconds)
    private val ALERT_EVERY_N_FRAMES = 36
    private var frameCount           = 0

    // FPS — tracked as a rolling 1-second window for on-screen display
    private var lastFpsTimestamp = 0L
    private var fpsFrameCount    = 0
    private var currentFps       = 0f

    // ── Lifecycle ──────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder  = findViewById(R.id.viewFinder)
        overlayView = findViewById(R.id.overlayView)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // 1. Copy YOLO model from assets to internal storage
        val modelPath = copyAssetToFile("yolov8n.onnx")

        // 2. Initialise ORT session (Java ORT API — no Python onnxruntime needed)
        ortEnv = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
        try {
            Log.d(TAG, "Attempting to enable NNAPI for NPU acceleration...")
            opts.addNnapi()
            ortSession = ortEnv.createSession(modelPath, opts)
            executionProvider = "NNAPI (NPU/GPU)"
            Log.d(TAG, "ORT session initialized with NNAPI (NPU).")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to initialize ONNX Runtime with NNAPI (NPU): ${e.message}. Falling back to CPU.")
            // Re-create options and session for CPU fallback
            val cpuOpts = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
            ortSession = ortEnv.createSession(modelPath, cpuOpts)
            executionProvider = "CPU (Fallback)"
            Log.d(TAG, "ORT session initialized with CPU fallback.")
        }
        ortInputName = ortSession.inputNames.iterator().next()
        Log.d(TAG, "ORT session ready. Input: $ortInputName")

        if (!Python.isStarted()) Python.start(AndroidPlatform(this))
        pyModule = Python.getInstance().getModule("camera_android")
        Log.d(TAG, "Python module loaded")

        // 5. Initialise Android TTS
        tts = TextToSpeech(this, this)

        // 6. Request camera permission
        checkAndRequestCameraPermission()
    }

    /** TextToSpeech.OnInitListener */
    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts?.setLanguage(Locale.US)
            ttsReady = (result != TextToSpeech.LANG_MISSING_DATA &&
                        result != TextToSpeech.LANG_NOT_SUPPORTED)
            tts?.setSpeechRate(1.1f)   // Slightly faster for navigation
            if (ttsReady) Log.d(TAG, "TTS ready")
            else          Log.w(TAG, "TTS language not supported")
        } else {
            Log.e(TAG, "TTS init failed with status $status")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        tts?.stop()
        tts?.shutdown()
        cameraExecutor.shutdown()
        if (::ortSession.isInitialized) ortSession.close()
        if (::ortEnv.isInitialized)     ortEnv.close()
    }

    private fun computeAndSetFocalLength(frameW: Int, frameH: Int) {
        if (focalLengthSet) return

        if (HARDCODED_FOCAL_LENGTH_PX != null && HARDCODED_FOCAL_LENGTH_PX > 0f) {
            Log.d(TAG, "Using manual hardcoded focal length: ${HARDCODED_FOCAL_LENGTH_PX}px")
            pyModule.callAttr("set_focal_length", HARDCODED_FOCAL_LENGTH_PX.toDouble())
            focalLengthSet = true
            return
        }

        val sharedPrefs = getSharedPreferences("BlindNavPrefs", MODE_PRIVATE)
        val saved = sharedPrefs.getFloat("focal_length_px_v3", -1f)
        if (saved > 0f) {
            Log.d(TAG, "Loaded saved focal length: ${String.format("%.2f", saved)}px")
            pyModule.callAttr("set_focal_length", saved.toDouble())
            focalLengthSet = true
            Log.i(TAG, """
                
                ==================================================
                   BLIND NAVIGATION APP INITIALIZED SUCCESSFUL
                ==================================================
                 * Execution Provider : $executionProvider
                 * Focal Length       : ${String.format("%.2f", saved)} px (Saved)
                 * Target Frame Rate  : $TARGET_FPS FPS
                ==================================================
            """.trimIndent())
            return
        }

        try {
            val cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
            val cameraId = cameraManager.cameraIdList.firstOrNull { id ->
                cameraManager.getCameraCharacteristics(id)
                    .get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
            } ?: return

            val chars = cameraManager.getCameraCharacteristics(cameraId)
            
            val focalMm = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: return
            val sensorSize = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE) ?: return
            val activeArray = chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE) ?: return

            // Fix: Scale using the long edge of the frame to match the sensor's long edge (width).
            // In portrait mode, frameH is the long edge (e.g., 640px). 
            val sensorWidthMm = sensorSize.width
            val activeWidthPx = activeArray.width()
            val frameLongEdge = if (frameW > frameH) frameW else frameH
            val scale = frameLongEdge.toFloat() / activeWidthPx
            val focalPx = focalMm * (activeWidthPx.toFloat() / sensorWidthMm) * scale

            Log.d(TAG, "Improved focal calc: ${String.format("%.2f", focalPx)} px")

            sharedPrefs.edit().putFloat("focal_length_px_v3", focalPx).apply()
            pyModule.callAttr("set_focal_length", focalPx.toDouble())
            focalLengthSet = true

            Log.i(TAG, """
                
                ==================================================
                   BLIND NAVIGATION APP INITIALIZED SUCCESSFUL
                ==================================================
                 * Execution Provider : $executionProvider
                 * Focal Length       : ${String.format("%.2f", focalPx)} px (Calculated)
                 * Target Frame Rate  : $TARGET_FPS FPS
                ==================================================
            """.trimIndent())

        } catch (e: Exception) {
            Log.e(TAG, "Focal length calculation failed", e)
            val fallback = 1400f
            pyModule.callAttr("set_focal_length", fallback.toDouble())
            focalLengthSet = true
        }
    }

    // ── Camera permission ──────────────────────────────────────────────────────

    private fun checkAndRequestCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_REQUEST_CODE)
        } else {
            startCamera()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED)
                startCamera()
            else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }

    // ── Camera setup ───────────────────────────────────────────────────────────

    private fun startCamera() {
        ProcessCameraProvider.getInstance(this).addListener({
            val cameraProvider = ProcessCameraProvider.getInstance(this).get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                processFrame(imageProxy)
            }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this as androidx.lifecycle.LifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview, imageAnalysis
                )
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ── Frame processing ───────────────────────────────────────────────────────

    private fun processFrame(imageProxy: ImageProxy) {
        // ── 12 FPS cap: drop frames that arrive faster than FRAME_INTERVAL_MS ──
        val now0 = System.currentTimeMillis()
        if (now0 - lastFrameTimestamp < FRAME_INTERVAL_MS) {
            imageProxy.close()
            return
        }
        lastFrameTimestamp = now0

        try {
            val rotDeg  = imageProxy.imageInfo.rotationDegrees
            val rawW    = imageProxy.width
            val rawH    = imageProxy.height
            val frameW  = if (rotDeg == 90 || rotDeg == 270) rawH else rawW
            val frameH  = if (rotDeg == 90 || rotDeg == 270) rawW else rawH

            // Dynamically calibrate focal length on first frame
            if (!focalLengthSet) {
                computeAndSetFocalLength(frameW, frameH)
            }

            // ── Step 1: Build input tensor ─────────────────────────────────────
            val inputBuf = imageProxyToFloatBuffer(imageProxy, rotDeg)
            val shape    = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
            val tensor   = OnnxTensor.createTensor(ortEnv, inputBuf, shape)

            // ── Step 2: ONNX inference (Kotlin/Java) ───────────────────────────
            val ortResults = ortSession.run(mapOf(ortInputName to tensor))
            @Suppress("UNCHECKED_CAST")
            val outputArr = ortResults[0].value as Array<Array<FloatArray>>
            // YOLOv8 output: (1, 84, 8400)
            val numRows = outputArr[0].size       // 84
            val numCols = outputArr[0][0].size    // 8400

            // Scale boxes from model-input space → frame pixel space
            val scaleX = frameW.toFloat() / INPUT_SIZE
            val scaleY = frameH.toFloat() / INPUT_SIZE
            val flatList = ArrayList<Float>(numRows * numCols)
            for (row in outputArr[0]) for (v in row) flatList.add(v)
            for (col in 0 until numCols) {
                flatList[0 * numCols + col] *= scaleX
                flatList[1 * numCols + col] *= scaleY
                flatList[2 * numCols + col] *= scaleX
                flatList[3 * numCols + col] *= scaleY
            }

            tensor.close(); ortResults.close()

            // ── Step 3: Python — detection postprocessing ──────────────────────
            val pyDetections = pyModule.callAttr(
                "postprocess_output",
                flatList.toFloatArray(), numRows, numCols, frameW, frameH
            )

            // ── Step 4: Python — distance estimation (Phase 2) ─────────────────
            val pyEstimates = pyModule.callAttr(
                "estimate_distances", pyDetections, frameW, frameH
            )

            // ── Step 5: Python — clipping check (Phase 2) ──────────────────────
            val pyClipping = pyModule.callAttr("check_clipping", pyEstimates, frameH)

            // ── Step 6: Python — zone status for overlay ────────────────────────
            val pyZoneStatus = pyModule.callAttr("zone_status", pyEstimates)

            // ── Step 7: Python — alert text (rate-limited) ──────────────────────
            frameCount++
            var pyAlerts: PyObject? = null
            if (frameCount % ALERT_EVERY_N_FRAMES == 0) {
                pyAlerts = pyModule.callAttr("format_alerts", pyEstimates, pyClipping)
            }

            // ── Step 8: Speak alerts via Android TTS ────────────────────────────
            pyAlerts?.asList()?.forEach { alertObj ->
                val alertMap  = alertObj.asMap()
                val text      = alertMap[PyObject.fromJava("text")]!!.toString()
                val isPriority = alertMap[PyObject.fromJava("priority")]!!.toBoolean()
                speakAlert(text, isPriority)
            }

            // ── Step 9: Map estimates → Kotlin Detection objects ─────────────────
            val detectionsList = mutableListOf<OverlayView.Detection>()
            for (obj in pyEstimates.asList()) {
                val m         = obj.asMap()
                val label     = m[PyObject.fromJava("label")]!!.toString()
                val score     = m[PyObject.fromJava("score")]!!.toFloat()
                val isNav     = m[PyObject.fromJava("is_nav_relevant")]!!.toBoolean()
                val boxPy     = m[PyObject.fromJava("box")]!!.asList()
                val box       = intArrayOf(
                    boxPy[0].toInt(), boxPy[1].toInt(),
                    boxPy[2].toInt(), boxPy[3].toInt()
                )
                val distPy    = m[PyObject.fromJava("distance")]
                val distance: Float? = try { distPy?.toFloat() } catch (e: Exception) { null }
                val zone      = m[PyObject.fromJava("zone")]!!.toString()
                detectionsList.add(OverlayView.Detection(label, score, box, isNav, distance, zone))
            }

            // Parse zone status
            val zsMap      = pyZoneStatus.asMap()
            val zoneStatus = zsMap[PyObject.fromJava("status")]!!.toString()
            val zoneDistPy = zsMap[PyObject.fromJava("distance")]
            val zoneDist: Float? = try { zoneDistPy?.toFloat() } catch (e: Exception) { null }

            // Extract clipping as a plain Kotlin Boolean
            val clippingList  = pyClipping.asList()
            val isClippingNow = clippingList.isNotEmpty()

            // ── Step 10: Update UI ───────────────────────────────────────────────
            runOnUiThread {
                overlayView.setResults(
                    detectionsList, frameW, frameH,
                    zoneStatus, zoneDist,
                    isClippingNow, currentFps
                )
            }

            // ── FPS — rolling 1-second window, shown on screen ───────────────────
            fpsFrameCount++
            val now = System.currentTimeMillis()
            if (now - lastFpsTimestamp >= 1000) {
                currentFps       = fpsFrameCount.toFloat() * 1000f / (now - lastFpsTimestamp)
                Log.d(TAG, "FPS: ${String.format("%.1f", currentFps)}")
                fpsFrameCount    = 0
                lastFpsTimestamp = now
            }

        } catch (e: Exception) {
            Log.e(TAG, "processFrame error", e)
        } finally {
            imageProxy.close()
        }
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private fun copyAssetToFile(name: String): String {
        val file = File(filesDir, name)
        if (!file.exists()) {
            assets.open(name).use { i -> FileOutputStream(file).use { o ->
                val buf = ByteArray(4096); var n: Int
                while (i.read(buf).also { n = it } != -1) o.write(buf, 0, n)
                o.flush()
            }}
        }
        return file.absolutePath
    }

    private fun imageProxyToFloatBuffer(imageProxy: ImageProxy, rotDeg: Int): FloatBuffer {
        val bitmap  = imageProxy.toBitmap()
        val rotated = if (rotDeg == 0) bitmap
                      else Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height,
                               Matrix().apply { postRotate(rotDeg.toFloat()) }, true)
        val scaled  = Bitmap.createScaledBitmap(rotated, INPUT_SIZE, INPUT_SIZE, true)

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val ch = INPUT_SIZE * INPUT_SIZE
        val floatArray = FloatArray(3 * ch)
        for (i in 0 until ch) {
            val p = pixels[i]
            floatArray[i]          = ((p shr 16) and 0xFF) / 255f  // R
            floatArray[ch + i]     = ((p shr 8)  and 0xFF) / 255f  // G
            floatArray[2 * ch + i] = (p          and 0xFF) / 255f  // B
        }
        return FloatBuffer.wrap(floatArray)
    }

    private fun speakAlert(text: String, priority: Boolean) {
        if (!ttsReady) return
        val queueMode = if (priority) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
        tts?.speak(text, queueMode, null, text.hashCode().toString())
    }
}
