import { useEffect, useMemo, useRef, useState, type ChangeEvent, type FormEvent } from "react";
import { Rnd } from "react-rnd";
import type { DraggableData, DraggableEvent } from "react-draggable";
import type { ResizeDirection } from "re-resizable";
import "../css/videoChat.css";

type CameraRole = "main" | "leftWrist" | "rightWrist" | "other";

type CameraLayout = {
  x: number;
  y: number;
  width: number;
  height: number;
  zIndex: number;
};

const VideoChat = () => {
  const remoteVideo = useRef<HTMLVideoElement | null>(null);
  const unnamedTrackCounter = useRef<number>(0);
  const [hostIP, setHostIP] = useState<string>("");
  const [sessionID, setSessionID] = useState<string>("");
  const [connectionStatus, setConnectionStatus] = useState<
    "idle" | "connecting" | "connected" | "error"
  >("idle");
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [pc, setPc] = useState<RTCPeerConnection | null>(null);
  const [remoteStreams, setRemoteStreams] = useState<Record<string, MediaStream>>({});
  const [cameraLayouts, setCameraLayouts] = useState<Record<string, CameraLayout>>({});
  const zCounterRef = useRef<number>(10);

  const createPeerConnection = (): RTCPeerConnection => {
    const config: RTCConfiguration = {
      iceServers: [
        { urls: "stun:stun.l.google.com:19302" },
      ],
    };

    const newPc = new RTCPeerConnection(config);

    newPc.ontrack = (event) => {
      if (event.track.kind === "video") {
        const generatedTrackId = `video-view-${unnamedTrackCounter.current++}`;
        const trackId = event.track.id || generatedTrackId;
        const viewName = trackId.startsWith("video-")
          ? trackId.slice("video-".length)
          : trackId;
        const stream = event.streams[0];
        setRemoteStreams((prev: Record<string, MediaStream>) => ({
          ...prev,
          [viewName]: stream,
        }));
      }
    };

    newPc.oniceconnectionstatechange = () => {
      console.log("ICE connection state:", newPc.iceConnectionState);
      if (
        newPc.iceConnectionState === "disconnected" ||
        newPc.iceConnectionState === "failed" ||
        newPc.iceConnectionState === "closed"
      ) {
        // When the connection drops, update the UI accordingly.
        setErrorMessage("Stream Disconnected.");
        setConnectionStatus("error");
        setIsConnected(false);
        setRemoteStreams({});
        // Stop any active video tracks and clear the video element.
        if (remoteVideo.current && remoteVideo.current.srcObject) {
          const stream = remoteVideo.current.srcObject as MediaStream;
          stream.getTracks().forEach((track) => track.stop());
          remoteVideo.current.srcObject = null;
        }
      } else if (newPc.iceConnectionState === "connected") {
        setIsConnected(true);
      }
    };

    newPc.onicecandidate = (event) => {
      if (event.candidate) {
        console.log("New ICE candidate:", event.candidate);
      }
    };

    return newPc;
  };

  const startStream = async () => {
    if (isConnected || connectionStatus === "connecting") return;

    setConnectionStatus("connecting");
    setErrorMessage("");
    try {
      const newPc = createPeerConnection();
      setPc(newPc);

      // Request an offer from the server.
      const offerResponse = await fetch(`http://${hostIP}:8080/media/offer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: sessionID }),
      });

      if (!offerResponse.ok) {
        const errorText = await offerResponse.text();
        setErrorMessage(errorText || "Failed to fetch offer.");
        setConnectionStatus("error");
        return;
      }

      const offer = await offerResponse.json();
      console.log("Offer received:", offer);

      // Set the offer as remote description.
      await newPc.setRemoteDescription(new RTCSessionDescription(offer));

      // Create an answer.
      const answer = await newPc.createAnswer();
      await newPc.setLocalDescription(answer);

      // Send the answer to the server.
      const answerData = {
        sessionId: sessionID,
        sdp: answer.sdp,
        type: answer.type,
      };

      const answerResponse = await fetch(`http://${hostIP}:8080/media/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(answerData),
      });

      if (!answerResponse.ok) {
        const errorText = await answerResponse.text();
        setErrorMessage(errorText || "Failed to send answer.");
        setConnectionStatus("error");
        return;
      }

      // Connection successfully established.
      setIsConnected(true);
      setConnectionStatus("connected");
    } catch (error: unknown) {
      console.error("Error starting stream:", error);
      if (error instanceof Error) {
        setErrorMessage(error.message || "An error occurred while connecting.");
      } else {
        setErrorMessage("An error occurred while connecting.");
      }
      setConnectionStatus("error");
    }
  };

  const disconnectStream = () => {
    if (pc) {
      pc.close();
      setPc(null);
    }
    setIsConnected(false);
    setConnectionStatus("idle");
    setRemoteStreams({});
    if (remoteVideo.current && remoteVideo.current.srcObject) {
      const stream = remoteVideo.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      remoteVideo.current.srcObject = null;
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    await startStream();
  };

  const orderedViews = Object.keys(remoteStreams).sort((a, b) => {
    return a.localeCompare(b);
  });

  const pickMainView = (views: string[]): string | undefined => {
    const preferredMain = ["agentview", "front", "tiled_camera", "primary"];
    for (const preferred of preferredMain) {
      const match = views.find((view) => view.includes(preferred));
      if (match) return match;
    }
    return views[0];
  };

  const primaryView = pickMainView(orderedViews);

  const isWristView = (viewName: string): boolean => {
    return (
      viewName.includes("wrist") ||
      viewName.includes("eye_in_hand") ||
      viewName.includes("hand")
    );
  };

  const isLeftWristView = (viewName: string): boolean => {
    return viewName.includes("left") || viewName.includes("robot0");
  };

  const isRightWristView = (viewName: string): boolean => {
    return viewName.includes("right") || viewName.includes("robot1");
  };

  const nonPrimaryViews = orderedViews.filter((v) => v !== primaryView);
  const wristViews = nonPrimaryViews.filter(isWristView);
  const sideLeftView = wristViews.find(isLeftWristView);
  const sideRightView = wristViews.find(isRightWristView);
  const usedWristViews = new Set([sideLeftView, sideRightView].filter(Boolean));
  const remainingViews = nonPrimaryViews.filter((v) => !usedWristViews.has(v));

  const defaultLayoutForRole = (role: CameraRole, index: number): CameraLayout => {
    if (role === "main") {
      return { x: 270, y: 20, width: 860, height: 520, zIndex: 1 };
    }
    if (role === "leftWrist") {
      return { x: 20, y: 560, width: 360, height: 240, zIndex: 2 };
    }
    if (role === "rightWrist") {
      return { x: 1020, y: 560, width: 360, height: 240, zIndex: 2 };
    }
    return {
      x: 20 + index * 40,
      y: 20 + index * 30,
      width: 320,
      height: 220,
      zIndex: 3,
    };
  };

  const viewLabel = (viewName: string): string => {
    if (viewName.includes("wrist")) return "Wrist";
    return viewName.replaceAll("_", " ").replace(/\b\w/g, (ch) => ch.toUpperCase());
  };

  const cameraCards = useMemo(
    () => [
      ...(primaryView ? [{ viewName: primaryView, role: "main" as const, label: "Main View" }] : []),
      ...(sideLeftView
        ? [{ viewName: sideLeftView, role: "leftWrist" as const, label: "Left Wrist" }]
        : []),
      ...(sideRightView
        ? [{ viewName: sideRightView, role: "rightWrist" as const, label: "Right Wrist" }]
        : []),
      ...remainingViews.map((viewName) => ({
        viewName,
        role: "other" as const,
        label: viewLabel(viewName),
      })),
    ],
    [primaryView, remainingViews, sideLeftView, sideRightView]
  );

  useEffect(() => {
    setCameraLayouts((prev) => {
      const next: Record<string, CameraLayout> = { ...prev };
      let changed = false;

      cameraCards.forEach((card, idx) => {
        if (!next[card.viewName]) {
          next[card.viewName] = defaultLayoutForRole(card.role, idx);
          changed = true;
        }
      });

      Object.keys(next).forEach((viewName) => {
        if (!cameraCards.some((card) => card.viewName === viewName)) {
          delete next[viewName];
          changed = true;
        }
      });

      return changed ? next : prev;
    });
  }, [cameraCards]);

  const bringToFront = (viewName: string) => {
    zCounterRef.current += 1;
    const nextZ = zCounterRef.current;
    setCameraLayouts((layouts) => ({
      ...layouts,
      [viewName]: {
        ...layouts[viewName],
        zIndex: nextZ,
      },
    }));
  };

  return (
    <div className="video-container">
      <div className="video-stage">
        {cameraCards.length === 0 && connectionStatus === "idle" && (
          <div className="video-stage-empty">Waiting for camera streams...</div>
        )}

        {cameraCards.map((card, index) => {
          const stream = remoteStreams[card.viewName];
          if (!stream) return null;

          const layout =
            cameraLayouts[card.viewName] ?? defaultLayoutForRole(card.role, index);

          return (
            <Rnd
              key={card.viewName}
              bounds="parent"
              size={{ width: layout.width, height: layout.height }}
              position={{ x: layout.x, y: layout.y }}
              onDragStop={(_e: DraggableEvent, data: DraggableData) => {
                setCameraLayouts((prev) => ({
                  ...prev,
                  [card.viewName]: {
                    ...prev[card.viewName],
                    x: data.x,
                    y: data.y,
                  },
                }));
              }}
              onResizeStop={(
                _e: MouseEvent | TouchEvent,
                _direction: ResizeDirection,
                ref: HTMLElement,
                _delta: { height: number; width: number },
                position: { x: number; y: number }
              ) => {
                setCameraLayouts((prev) => ({
                  ...prev,
                  [card.viewName]: {
                    ...prev[card.viewName],
                    x: position.x,
                    y: position.y,
                    width: parseInt(ref.style.width, 10),
                    height: parseInt(ref.style.height, 10),
                  },
                }));
              }}
              onMouseDown={() => bringToFront(card.viewName)}
              minWidth={card.role === "main" ? 420 : 240}
              minHeight={card.role === "main" ? 260 : 170}
              className={`camera-window camera-window-${card.role}`}
              style={{ zIndex: layout.zIndex }}
              resizeHandleClasses={{
                top: "resize-handle resize-handle-top",
                right: "resize-handle resize-handle-right",
                bottom: "resize-handle resize-handle-bottom",
                left: "resize-handle resize-handle-left",
                topRight: "resize-handle resize-handle-top-right",
                topLeft: "resize-handle resize-handle-top-left",
                bottomRight: "resize-handle resize-handle-bottom-right",
                bottomLeft: "resize-handle resize-handle-bottom-left",
              }}
              resizeHandleStyles={{
                top: { height: "12px", top: "-6px" },
                right: { width: "12px", right: "-6px" },
                bottom: { height: "12px", bottom: "-6px" },
                left: { width: "12px", left: "-6px" },
                topRight: { width: "20px", height: "20px", top: "-10px", right: "-10px" },
                topLeft: { width: "20px", height: "20px", top: "-10px", left: "-10px" },
                bottomRight: { width: "20px", height: "20px", bottom: "-10px", right: "-10px" },
                bottomLeft: { width: "20px", height: "20px", bottom: "-10px", left: "-10px" },
              }}
            >
              <div className={`camera-window-label ${card.role === "rightWrist" ? "camera-window-label-right" : ""}`} >
                {card.label}
              </div>
              <video
                ref={(el: HTMLVideoElement | null) => {
                  if (el && el.srcObject !== stream) {
                    el.srcObject = stream;
                  }
                  if (card.role === "main") {
                    remoteVideo.current = el;
                  }
                }}
                autoPlay
                playsInline
                className="camera-window-video"
              />
              <div className="camera-window-resize-cue" aria-hidden="true" />
            </Rnd>
          );
        })}

        {(connectionStatus === "connecting" || connectionStatus === "error") && (
          <div className="overlay">
            {connectionStatus === "connecting" && (
              <>
                <div className="spinner"></div>
                <div className="status-message">Connecting...</div>
              </>
            )}
            {connectionStatus === "error" && (
              <div className="error-message">
                <div>{errorMessage}</div>
                <div className="try-again">Try Again</div>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="settings-box">
        <form className="video-settings" onSubmit={handleSubmit}>
          <div className="input-group">
            <label htmlFor="hostIP" className="video-label">
              Host IP:
            </label>
            <input
              type="text"
              id="hostIP"
              value={hostIP}
              onChange={(e: ChangeEvent<HTMLInputElement>) =>
                setHostIP(e.target.value)
              }
              className="video-input"
              placeholder="e.g., 192.168.1.100"
            />
          </div>
          <div className="input-group">
            <label htmlFor="sessionID" className="video-label">
              Session ID:
            </label>
            <input
              type="text"
              id="sessionID"
              value={sessionID}
              onChange={(e: ChangeEvent<HTMLInputElement>) =>
                setSessionID(e.target.value)
              }
              className="video-input"
              placeholder="Enter your session ID"
            />
          </div>
          <div className="button-group">
            <button
              type="submit"
              disabled={connectionStatus === "connecting" || isConnected}
              className="video-button"
            >
              {isConnected ? "Connected" : "Start Stream"}
            </button>
            {isConnected && (
              <button
                type="button"
                onClick={disconnectStream}
                className="video-button disconnect-button"
              >
                Disconnect
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
};

export default VideoChat;
