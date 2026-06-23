package com.example.blindnav

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View

/**
 * OverlayView — mirrors run_phase2.py output exactly:
 *
 *  1. Bounding box   — 2px stroke, per-class colour  (bbox_renderer.py: render())
 *  2. Label badge    — "person 0.87" — coloured bg, BLACK text  (bbox_renderer.py)
 *  3. Distance text  — "1.8m left"   — YELLOW above the label   (draw_distances())
 *  4. FPS counter    — "FPS: 14.2"   — GREEN top-left           (fps overlay)
 *  5. Zone border    — 6px frame border coloured by zone        (draw_zone_indicator())
 *  6. Zone bar       — 50dp filled bottom bar + white status text
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    data class Detection(
        val label: String,
        val score: Float,
        val box: IntArray,            // [x1, y1, x2, y2] in frame pixels
        val isNavRelevant: Boolean,
        val distance: Float?,         // metres, null if unknown
        val zone: String              // "left" | "centre" | "right"
    )

    // ── State ──────────────────────────────────────────────────────────────────
    private var results:    List<Detection> = emptyList()
    private var srcWidth:   Int    = 1
    private var srcHeight:  Int    = 1
    private var zoneStatus: String = "clear"   // "danger"|"caution"|"safe"|"clipping"|"clear"
    private var zoneDist:   Float? = null
    private var isClipping: Boolean = false
    private var currentFps: Float   = 0f

    // ── Zone colours (BGR→RGB from OpenCV run_phase2.py) ──────────────────────
    //  cv2 danger  = (0,0,255) BGR  → rgb(255,0,0) red
    //  cv2 caution = (0,140,255) BGR → rgb(255,140,0) orange
    //  cv2 safe    = (0,200,80) BGR  → rgb(80,200,0) green
    private val COLOR_DANGER  = Color.rgb(220,  30,  30)
    private val COLOR_CAUTION = Color.rgb(255, 140,   0)
    private val COLOR_SAFE    = Color.rgb( 30, 200,  30)

    // ── Per-class colours (BGR→RGB from bbox_renderer.py CLASS_COLORS) ────────
    //  Note: bbox_renderer uses BGR, Android uses RGB — swap R and B
    private val classColors = mapOf(
        "person"       to Color.rgb(120, 255,   0),  // (0,255,120) BGR → RGB
        "bicycle"      to Color.rgb(255, 220,   0),  // (0,220,255)
        "car"          to Color.rgb(255,  60,   0),  // (0,60,255)
        "motorcycle"   to Color.rgb(255, 140,   0),  // (0,140,255)
        "bus"          to Color.rgb(255,   0,   0),  // (0,0,255)
        "truck"        to Color.rgb(200,   0,   0),  // (0,0,200)
        "chair"        to Color.rgb(  0, 180, 255),  // (255,180,0)
        "couch"        to Color.rgb( 20, 160, 255),  // (255,160,20)
        "bed"          to Color.rgb( 40, 140, 255),  // (255,140,40)
        "dining table" to Color.rgb( 60, 120, 255),  // (255,120,60)
        "toilet"       to Color.rgb(200, 100, 200),  // (200,100,200)
    )
    private val colorDefault = Color.rgb(80, 80, 80)   // DEFAULT_COLOR

    // ── Box paint — 2px stroke matching cv2.rectangle thickness=2 ─────────────
    private val paintBox = Paint().apply {
        style       = Paint.Style.STROKE
        strokeWidth = 4f           // ~2 logical px * density
        isAntiAlias = true
    }
    // Filled rect for label background
    private val paintFill = Paint().apply { style = Paint.Style.FILL }

    // ── Label text: "person 0.87" — black, matching cv2 black text ────────────
    private val paintLabel = Paint().apply {
        color       = Color.BLACK
        textSize    = 36f
        isAntiAlias = true
        typeface    = Typeface.DEFAULT
    }

    // ── Distance text: "1.8m left" — YELLOW (255,255,0) ──────────────────────
    private val paintDist = Paint().apply {
        color       = Color.rgb(255, 255, 0)   // exact match to (255,255,0) cv2 colour
        textSize    = 32f
        isAntiAlias = true
        typeface    = Typeface.DEFAULT
    }

    // ── FPS text: "FPS: 14.2" — GREEN (0,255,0) at top-left ─────────────────
    private val paintFps = Paint().apply {
        color       = Color.rgb(0, 255, 0)
        textSize    = 52f          // cv2 fontScale 1.0 ≈ large text on screen
        isAntiAlias = true
        typeface    = Typeface.DEFAULT_BOLD
    }

    // ── Zone border + bar ─────────────────────────────────────────────────────
    private val paintBorder = Paint().apply {
        style       = Paint.Style.STROKE
        strokeWidth = 12f          // cv2 thickness=6 → ~12 logical px
        isAntiAlias = false
    }
    private val paintBar   = Paint().apply { style = Paint.Style.FILL }
    private val paintBarText = Paint().apply {
        color       = Color.WHITE
        textSize    = 46f          // cv2 fontScale 0.8 → roughly 46sp
        isAntiAlias = true
        typeface    = Typeface.DEFAULT_BOLD
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    fun setResults(
        detections:  List<Detection>,
        srcWidth:    Int,
        srcHeight:   Int,
        zoneStatus:  String,
        zoneDist:    Float?,
        isClipping:  Boolean,
        fps:         Float
    ) {
        this.results    = detections
        this.srcWidth   = srcWidth
        this.srcHeight  = srcHeight
        this.zoneStatus = zoneStatus
        this.zoneDist   = zoneDist
        this.isClipping = isClipping
        this.currentFps = fps
        postInvalidate()
    }

    // ── Drawing ────────────────────────────────────────────────────────────────

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (srcWidth <= 0 || srcHeight <= 0) return

        val scaleX = width.toFloat()  / srcWidth
        val scaleY = height.toFloat() / srcHeight

        // 1. Bounding boxes + label badges + distance text
        //    Order matches run_phase2.py: render() then draw_distances()
        for (det in results) {
            val color = classColors[det.label] ?: colorDefault
            paintBox.color  = color
            paintFill.color = color

            val x1 = det.box[0] * scaleX
            val y1 = det.box[1] * scaleY
            val x2 = det.box[2] * scaleX
            val y2 = det.box[3] * scaleY

            // ── Box (bbox_renderer: cv2.rectangle thickness=2) ─────────────────
            canvas.drawRect(x1, y1, x2, y2, paintBox)

            // ── Label badge: "person 0.87" — coloured bg, black text ───────────
            val labelText = "${det.label} ${String.format("%.2f", det.score)}"
            val lw = paintLabel.measureText(labelText)
            val lh = -paintLabel.fontMetrics.ascent + paintLabel.fontMetrics.descent
            // background rect above box  (x1, y1-th-8) → (x1+tw+4, y1)
            canvas.drawRect(x1, y1 - lh - 8f, x1 + lw + 8f, y1, paintFill)
            canvas.drawText(labelText, x1 + 4f, y1 - paintLabel.fontMetrics.descent - 4f, paintLabel)

            // ── Distance text above label: "1.8m left" — yellow ───────────────
            //    draw_distances: cv2.putText at (x1, y1-20) — above the label
            if (det.distance != null) {
                val distText = "${det.distance}m ${det.zone}"
                // Position above the label badge (y1 - lh - 8 - small margin)
                val distY = y1 - lh - 16f
                canvas.drawText(distText, x1, distY, paintDist)
            }
        }

        // 2. Zone indicator: border + bottom bar
        //    Matches draw_zone_indicator() in run_phase2.py exactly
        val zoneColor: Int?
        val statusText: String?

        if (isClipping) {
            // clipping case: "DANGER  {closest}m  TOO CLOSE"
            val closest = results.firstOrNull { it.distance != null && (it.distance > 0) }?.distance
            zoneColor  = COLOR_DANGER
            statusText = if (closest != null) "DANGER  ${closest}m  TOO CLOSE" else "DANGER  TOO CLOSE"
        } else {
            when (zoneStatus) {
                "danger"  -> { zoneColor = COLOR_DANGER;  statusText = "DANGER   ${zoneDist}m" }
                "caution" -> { zoneColor = COLOR_CAUTION; statusText = "CAUTION  ${zoneDist}m" }
                "safe"    -> { zoneColor = COLOR_SAFE;    statusText = "SAFE     ${zoneDist}m" }
                else      -> { zoneColor = null;          statusText = null }
            }
        }

        if (zoneColor != null && statusText != null) {
            // 6px border around full frame
            paintBorder.color = zoneColor
            canvas.drawRect(6f, 6f, width - 6f, height - 6f, paintBorder)

            // 50dp bottom bar
            val barH = 100f   // ~50dp * 2 density
            paintBar.color = zoneColor
            canvas.drawRect(0f, height - barH, width.toFloat(), height.toFloat(), paintBar)
            canvas.drawText(statusText, 20f,
                height - barH + barH - paintBarText.fontMetrics.descent - 12f, paintBarText)
        }

        // 3. FPS counter — "FPS: 14.2" — green top-left
        //    Matches: cv2.putText(frame, f"FPS: {fps.get():.1f}", (10,30), ..., scale=1.0, (0,255,0))
        if (currentFps > 0f) {
            val fpsText = "FPS: ${String.format("%.1f", currentFps)}"
            canvas.drawText(fpsText, 16f, 80f, paintFps)
        }
    }
}
