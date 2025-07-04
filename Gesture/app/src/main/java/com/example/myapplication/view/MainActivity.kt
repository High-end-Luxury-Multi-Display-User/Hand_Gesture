package com.example.myapplication.view

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.myapplication.R
import com.example.myapplication.ml.KeypointClassifier
import com.example.myapplication.ml.ModelMetadata
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer


class MainActivity : AppCompatActivity() {
    val permission = arrayOf("android.car.permission.CAR_VENDOR_EXTENSION")
    val steeringPermissionCode = 200
    private val CAMERA_PERMISSION_CODE = 100
    private lateinit var model: ModelMetadata
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var textureView: TextureView
    lateinit var result : TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        textureView =findViewById(R.id.textureView)

        result = findViewById(R.id.result)

        model = ModelMetadata.newInstance(this)
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        checkCameraPermission()
        setupTextureView()
        requestPermissions(permission, steeringPermissionCode)

    }

    private fun checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(
                this,
                android.Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(android.Manifest.permission.CAMERA),
                CAMERA_PERMISSION_CODE
            )
        } else {
            setupTextureView()
        }
    }

    private fun setupTextureView() {
        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(surfaceTexture: SurfaceTexture, width: Int, height: Int) {
                startCamera(surfaceTexture)
            }

            override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
            }

            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                return true
            }

            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
            }
        }
    }

    @SuppressLint("MissingPermission")
    private fun startCamera(surfaceTexture: SurfaceTexture) {
        val cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val cameraId = cameraManager.cameraIdList.first { id ->
            val characteristics = cameraManager.getCameraCharacteristics(id)
            characteristics.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
        }


        surfaceTexture?.setDefaultBufferSize(textureView.width, textureView.height)
        val surface = Surface(surfaceTexture)


        cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                val previewRequestBuilder = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                previewRequestBuilder.addTarget(surface)

                camera.createCaptureSession(listOf(surface), object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        session.setRepeatingRequest(previewRequestBuilder.build(), null, null)
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Log.e("CameraDebug", "Configuration failed")
                    }
                }, null)
                if (surfaceTexture != null) {
                    processFramesFromTexture()
                }
            }

            override fun onDisconnected(camera: CameraDevice) {
                camera.close()
            }

            override fun onError(camera: CameraDevice, error: Int) {
                Log.e("CameraDebug", "Camera error: $error")
            }
        }, null)
    }

    private fun processFramesFromTexture() {

        CoroutineScope(Dispatchers.Default).launch {
            while (true) {
                try {

                    val bitmap = textureView.bitmap ?: continue

                    var image = TensorImage.fromBitmap(bitmap)

                    image = imageProcessor.process(image)

                    val outputs = model.process(image)

                    val detectionResult = outputs.probabilityAsCategoryList

                    if (detectionResult.isNotEmpty()) {
                        val bestResult = detectionResult.maxByOrNull { it.score }

                        bestResult?.let {
                            if (it.score > 0.8f) {
                                Log.i("Gesture", "Label: ${it.label}")
                                Log.i("Gesture", "DisplayName: ${it.displayName}")
                                Log.i("Gesture", "Score: ${it.score}")
                                result.text = it.label
                            } else {
                                Log.i("Gesture", "No gesture confident enough (max score: ${it.score})")
                                result.text = "------"
                            }

                        }
                    }
                    delay(500)
                } catch (e: Exception) {
                    Log.e("TAG", "Frame processing error", e)
                }
            }
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        when(requestCode){
            CAMERA_PERMISSION_CODE ->{
                if (requestCode == CAMERA_PERMISSION_CODE && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    setupTextureView()
                } else {
                    Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show()
                }
            }
        }

    }

}