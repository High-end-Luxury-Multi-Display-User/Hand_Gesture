package android.vendor.camerawithservice

import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.util.Log
import android.vendor.camerawithservice.ml.ModelMetadata
import kotlinx.coroutines.*
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class CameraCaptureService : Service() {

    private val TAG = "GestureService"
    private lateinit var cameraManager: CameraManager
    private lateinit var imageReader: ImageReader
    private lateinit var gestureModel: ModelMetadata
    private lateinit var imageProcessor: ImageProcessor

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private lateinit var backgroundHandler: Handler
    private lateinit var backgroundThread: HandlerThread
    private val scope = CoroutineScope(Dispatchers.Default)

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "Service created ✅")

        gestureModel = ModelMetadata.newInstance(this)
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        startBackgroundThread()
        setupImageReader()
        openCamera()
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackgroundThread")
        backgroundThread.start()
        backgroundHandler = Handler(backgroundThread.looper)
    }

    private fun setupImageReader() {
        imageReader = ImageReader.newInstance(640, 480, ImageFormat.YUV_420_888, 2)
        imageReader.setOnImageAvailableListener({ reader ->
            val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener

            val bitmap = YuvToRgbConverter.yuvToRgb(this, image)
            image.close()

            scope.launch {
                try {
                    val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))
                    val result = gestureModel.process(tensorImage)
                    val best = result.probabilityAsCategoryList.maxByOrNull { it.score }
                    best?.let {
                        if (it.score > 0.8f) {
                            Log.i(TAG, "Gesture Detected: ${it.label} (${(it.score * 100).toInt()}%)")
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Inference failed", e)
                }
            }
        }, backgroundHandler)
    }

    private fun openCamera() {
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val cameraId = cameraManager.cameraIdList.first()

        if (checkSelfPermission(android.Manifest.permission.CAMERA) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
            Log.e(TAG, "Camera permission not granted")
            return
        }

        cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                cameraDevice = camera
                val surface = imageReader.surface
                val requestBuilder = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                requestBuilder.addTarget(surface)

                camera.createCaptureSession(listOf(surface), object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        captureSession = session
                        session.setRepeatingRequest(requestBuilder.build(), null, backgroundHandler)
                        Log.d(TAG, "Capture session started ✅")
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Log.e(TAG, "Capture session failed ❌")
                    }
                }, backgroundHandler)
            }

            override fun onDisconnected(camera: CameraDevice) = camera.close()
            override fun onError(camera: CameraDevice, error: Int) {
                Log.e(TAG, "Camera error: $error")
                camera.close()
            }
        }, backgroundHandler)
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
        gestureModel.close()
        imageReader.close()
        cameraDevice?.close()
        captureSession?.close()
        backgroundThread.quitSafely()
        Log.d(TAG, "Service destroyed ❌")
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
