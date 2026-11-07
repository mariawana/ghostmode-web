
import { useEffect, useRef, useState } from "react";
import {
  FilesetResolver,
  ImageSegmenter,
  HandLandmarker,
} from "@mediapipe/tasks-vision";
import "./App.css";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";

const SEGMENTER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite";

const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const backgroundRef = useRef(null);

  const segmenterRef = useRef(null);
  const handsRef = useRef(null);
  const streamRef = useRef(null);
  const animationRef = useRef(null);

  const invisibleRef = useRef(false);
  const previousPinchRef = useRef(false);
  const lastToggleRef = useRef(0);

  const smoothedMaskRef = useRef(null);
  const previousHandStatusRef = useRef("");

  const [isRunning, setIsRunning] = useState(false);
  const [isInvisible, setIsInvisible] = useState(false);
  const [hasBackground, setHasBackground] = useState(false);

  const [status, setStatus] = useState("Ready to start");
  const [handStatus, setHandStatus] = useState("Waiting for camera");
  const [error, setError] = useState("");

  async function initializeModels() {
    if (segmenterRef.current && handsRef.current) return;

    setStatus("Loading AI models...");

    const vision = await FilesetResolver.forVisionTasks(WASM_URL);

    const segmenter = await ImageSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: SEGMENTER_MODEL,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      outputCategoryMask: true,
    });

    let handLandmarker;

    try {
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: HAND_MODEL,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.3,
        minHandPresenceConfidence: 0.3,
        minTrackingConfidence: 0.3,
      });
    } catch (err) {
      console.warn("GPU hand detection unavailable. Trying CPU.", err);

      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: HAND_MODEL,
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.3,
        minHandPresenceConfidence: 0.3,
        minTrackingConfidence: 0.3,
      });
    }

    segmenterRef.current = segmenter;
    handsRef.current = handLandmarker;
  }

  function updateHandStatus(value) {
    if (previousHandStatusRef.current !== value) {
      previousHandStatusRef.current = value;
      setHandStatus(value);
    }
  }

  function toggleInvisible() {
    if (!backgroundRef.current) {
      setError("Capture your background first.");
      return;
    }

    const nextState = !invisibleRef.current;

    invisibleRef.current = nextState;
    setIsInvisible(nextState);
    setError("");

    setStatus(
      nextState ? "Invisibility activated" : "Invisibility disabled"
    );
  }

  function renderLoop() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const segmenter = segmenterRef.current;
    const hands = handsRef.current;

    if (!video || !canvas || !segmenter || !hands) return;

    const ctx = canvas.getContext("2d");

    const width = video.videoWidth;
    const height = video.videoHeight;

    if (!width || !height) {
      animationRef.current = requestAnimationFrame(renderLoop);
      return;
    }

    canvas.width = width;
    canvas.height = height;

    const maskCanvas = document.createElement("canvas");
    const smoothMaskCanvas = document.createElement("canvas");
    const foregroundCanvas = document.createElement("canvas");

    maskCanvas.width = width;
    maskCanvas.height = height;

    smoothMaskCanvas.width = width;
    smoothMaskCanvas.height = height;

    foregroundCanvas.width = width;
    foregroundCanvas.height = height;

    const maskCtx = maskCanvas.getContext("2d");
    const smoothMaskCtx = smoothMaskCanvas.getContext("2d");
    const foregroundCtx = foregroundCanvas.getContext("2d");

    function processFrame() {
      if (!streamRef.current) return;

      try {
        const now = performance.now();

        // Person segmentation
        const segmentation = segmenter.segmentForVideo(video, now);
        const categoryMask = segmentation.categoryMask;

        // Hand detection
        const handResult = hands.detectForVideo(video, now);
        const detectedHands = handResult.landmarks || [];

        let isPinching = false;

        if (detectedHands.length > 0) {
          updateHandStatus("Hand detected");

          for (const hand of detectedHands) {
            const thumb = hand[4];
            const indexFinger = hand[8];

            const distance = Math.hypot(
              thumb.x - indexFinger.x,
              thumb.y - indexFinger.y
            );

            if (distance < 0.12) {
              isPinching = true;
              break;
            }
          }

          if (isPinching) {
            updateHandStatus("Pinch detected!");
          }
        } else {
          updateHandStatus("No hand detected - show your hand");
        }

        // Toggle only when a new pinch begins
        const pinchStarted =
          isPinching && !previousPinchRef.current;

        if (
          pinchStarted &&
          backgroundRef.current &&
          now - lastToggleRef.current > 800
        ) {
          invisibleRef.current = !invisibleRef.current;
          setIsInvisible(invisibleRef.current);

          setStatus(
            invisibleRef.current
              ? "Invisibility activated by gesture"
              : "Invisibility disabled by gesture"
          );

          lastToggleRef.current = now;
        }

        previousPinchRef.current = isPinching;

        // Normal camera view
        if (!invisibleRef.current || !backgroundRef.current) {
          ctx.clearRect(0, 0, width, height);
          ctx.drawImage(video, 0, 0, width, height);
        } else if (categoryMask) {
          const maskWidth = categoryMask.width;
          const maskHeight = categoryMask.height;
          const maskData = categoryMask.getAsUint8Array();

          // Match mask dimensions
          if (
            maskCanvas.width !== maskWidth ||
            maskCanvas.height !== maskHeight
          ) {
            maskCanvas.width = maskWidth;
            maskCanvas.height = maskHeight;
          }

          if (
            !smoothedMaskRef.current ||
            smoothedMaskRef.current.length !== maskData.length
          ) {
            smoothedMaskRef.current = new Float32Array(
              maskData.length
            );
          }

          const smoothedMask = smoothedMaskRef.current;

          const imageData = maskCtx.createImageData(
            maskWidth,
            maskHeight
          );

          // Temporal smoothing
          for (let i = 0; i < maskData.length; i++) {
            const currentValue = maskData[i] > 0 ? 1 : 0;

            smoothedMask[i] =
              smoothedMask[i] * 0.80 +
              currentValue * 0.20;

            const alpha = Math.round(smoothedMask[i] * 255);
            const pixel = i * 4;

            imageData.data[pixel] = 255;
            imageData.data[pixel + 1] = 255;
            imageData.data[pixel + 2] = 255;
            imageData.data[pixel + 3] = alpha;
          }

          maskCtx.putImageData(imageData, 0, 0);

          // Smooth the mask edges
          smoothMaskCtx.clearRect(0, 0, width, height);
          smoothMaskCtx.filter = "blur(2px)";
          smoothMaskCtx.drawImage(
            maskCanvas,
            0,
            0,
            width,
            height
          );
          smoothMaskCtx.filter = "none";

          // Draw captured background
          ctx.clearRect(0, 0, width, height);
          ctx.drawImage(
            backgroundRef.current,
            0,
            0,
            width,
            height
          );

          // Extract the foreground person
          foregroundCtx.clearRect(0, 0, width, height);

          foregroundCtx.globalCompositeOperation = "source-over";
          foregroundCtx.drawImage(video, 0, 0, width, height);

          foregroundCtx.globalCompositeOperation = "destination-in";
          foregroundCtx.drawImage(
            smoothMaskCanvas,
            0,
            0,
            width,
            height
          );

          foregroundCtx.globalCompositeOperation = "source-over";

          // Composite person over captured background
          ctx.drawImage(foregroundCanvas, 0, 0);
        }

        categoryMask?.close();
      } catch (err) {
        console.error("Frame processing error:", err);
      }

      animationRef.current = requestAnimationFrame(processFrame);
    }

    processFrame();
  }

  async function startCamera() {
    try {
      setError("");
      setStatus("Starting camera...");
      setHandStatus("Loading hand detection...");

      smoothedMaskRef.current = null;
      previousPinchRef.current = false;

      await initializeModels();

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      streamRef.current = stream;

      const video = videoRef.current;
      video.srcObject = stream;

      await new Promise((resolve, reject) => {
        video.onloadedmetadata = resolve;
        video.onerror = reject;
      });

      await video.play();

      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      setIsRunning(true);
      setStatus("Camera is running");
      setHandStatus("Show your hand to the camera");

      animationRef.current = requestAnimationFrame(renderLoop);
    } catch (err) {
      console.error(err);

      setError(
        err.message || "Unable to start camera. Check camera permissions."
      );

      setStatus("Camera failed to start");
      stopCamera();
    }
  }

  function captureBackground() {
    const video = videoRef.current;

    if (!video || !streamRef.current) {
      setError("Start your camera first.");
      return;
    }

    const background = document.createElement("canvas");

    background.width = video.videoWidth;
    background.height = video.videoHeight;

    const ctx = background.getContext("2d");
    ctx.drawImage(video, 0, 0, background.width, background.height);

    backgroundRef.current = background;

    invisibleRef.current = false;
    smoothedMaskRef.current = null;
    previousPinchRef.current = false;

    setIsInvisible(false);
    setHasBackground(true);
    setError("");
    setStatus("Background captured successfully");
  }

  function takeScreenshot() {
    const canvas = canvasRef.current;

    if (!canvas || !streamRef.current) {
      setError("Start your camera first.");
      return;
    }

    const link = document.createElement("a");
    link.download = "ghostmode-screenshot.png";
    link.href = canvas.toDataURL("image/png");
    link.click();
  }

  function stopCamera() {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    segmenterRef.current?.close();
    handsRef.current?.close();

    segmenterRef.current = null;
    handsRef.current = null;

    backgroundRef.current = null;
    smoothedMaskRef.current = null;

    invisibleRef.current = false;
    previousPinchRef.current = false;
    previousHandStatusRef.current = "";

    setIsRunning(false);
    setIsInvisible(false);
    setHasBackground(false);
    setHandStatus("Waiting for camera");
    setStatus("Camera stopped");
  }

  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }

      streamRef.current?.getTracks().forEach((track) => track.stop());
      segmenterRef.current?.close();
      handsRef.current?.close();
    };
  }, []);

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          <span className="logo-icon">☀</span>
          <span>GHOSTMODE</span>
        </div>

        <span className="header-label">
          INVISIBILITY EXPERIENCE
        </span>
      </header>

      <main className="main">
        <section className="hero">
          <p className="eyebrow">YOUR CAMERA. YOUR MAGIC.</p>

          <h1>
            Become <span>Invisible.</span>
          </h1>

          <p className="hero-description">
            A little bit of technology, a little bit of magic.
            Make yourself disappear into your surroundings.
          </p>
        </section>

        <section className="camera-section">
          <div className="camera-container">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ display: "none" }}
            />

            <canvas
              ref={canvasRef}
              className="camera-canvas"
            />

            {!isRunning && (
              <div className="camera-placeholder">
                <div className="placeholder-icon">☀</div>
                <h2>Ready to disappear?</h2>
                <p>Start your camera to begin.</p>
              </div>
            )}
          </div>

          <div className="status">
            <span
              className={
                isRunning ? "status-dot active" : "status-dot"
              }
            />
            {status}
          </div>

          {isRunning && (
            <div
              style={{
                marginTop: "12px",
                padding: "10px 16px",
                background: "#202020",
                border: "1px solid #383838",
                borderRadius: "6px",
                color: "#ffd21f",
                fontSize: "12px",
                fontFamily: "monospace",
                textAlign: "center",
              }}
            >
              HAND DETECTION: {handStatus}
            </div>
          )}

          {error && (
            <p className="error-message">{error}</p>
          )}

          <div className="controls">
            {!isRunning ? (
              <button
                className="primary-button"
                onClick={startCamera}
              >
                Start Camera
              </button>
            ) : (
              <button
                className="secondary-button"
                onClick={stopCamera}
              >
                Stop Camera
              </button>
            )}

            <button
              className="secondary-button"
              onClick={captureBackground}
              disabled={!isRunning}
            >
              Capture Background
            </button>

            <button
              className="primary-button"
              onClick={toggleInvisible}
              disabled={!isRunning || !hasBackground}
            >
              {isInvisible ? "Appear" : "Disappear"}
            </button>

            <button
              className="secondary-button"
              onClick={takeScreenshot}
              disabled={!isRunning}
            >
              Screenshot
            </button>
          </div>
        </section>

        <section className="instructions">
          <div className="instruction-card">
            <span className="step-number">01</span>
            <h3>Start Camera</h3>
            <p>
              Allow camera access to see yourself on screen.
            </p>
          </div>

          <div className="instruction-card">
            <span className="step-number">02</span>
            <h3>Capture Background</h3>
            <p>
              Move out of the frame after capturing your background.
            </p>
          </div>

          <div className="instruction-card">
            <span className="step-number">03</span>
            <h3>Disappear</h3>
            <p>
              Click Disappear or pinch your thumb and index finger
              together.
            </p>
          </div>
        </section>

        <p className="gesture-note">
          ✦ Gesture control: Pinch your thumb and index finger
          together to toggle invisibility.
        </p>
      </main>

      <footer className="footer">
        GHOSTMODE · MADE WITH TECHNOLOGY
      </footer>
    </div>
  );
}

export default App;
