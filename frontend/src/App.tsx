import { useRef, useState, useEffect } from 'react';
import { startMediasoup, startStreaming } from './mediasoupClient';

function App() {
  const [joined, setJoined] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [roomCode, setRoomCode] = useState('a');
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [remoteAudioStreams, setRemoteAudioStreams] = useState<MediaStream[]>([]);
  const [remoteVideoStreams, setRemoteVideoStreams] = useState<MediaStream[]>([]);

  const localVideoRef = useRef<HTMLVideoElement>(null);

  const joinMeeting = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        //video: true,
        audio: true
      });
      // console.log(mediaStream.getVideoTracks()[0].getSettings());
      // const height = mediaStream.getVideoTracks()[0].getSettings().height;
      // const width = mediaStream.getVideoTracks()[0].getSettings().width;
      // localStorage.setItem('height', height!.toString());
      // localStorage.setItem('width', width!.toString());
      setStream(mediaStream);

      await startMediasoup(roomCode, (remoteStream, kind) => {
        if (kind === 'audio') {
          setRemoteAudioStreams((prev) => [...prev, remoteStream]);
        } else {
          setRemoteVideoStreams((prev) => [...prev, remoteStream]);
        }
      });

      await startStreaming(mediaStream, roomCode);
      setJoined(true);
    } catch (err: any) {
      console.trace('Join failed:', err);
      setError(err.message);
    }
  };

  useEffect(() => {
    if (joined && stream && localVideoRef.current) {
      localVideoRef.current.srcObject = new MediaStream(stream.getVideoTracks());
    }
  }, [joined, stream]);

  return (
    <div className="min-h-screen text-center py-12 px-4">
      {!joined && (
        <div className="space-y-4">
          <input
            type="text"
            placeholder="Enter meeting code"
            value={roomCode}
            onChange={(e) => setRoomCode(e.target.value)}
            className="px-4 py-2 border rounded-lg shadow-sm w-64 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <br />
          <button
            onClick={joinMeeting}
            className="px-6 py-2 bg-blue-600 rounded-lg hover:bg-blue-700 transition"
          >
            Join
          </button>
        </div>
      )}

      {error && <p className="text-red-600 mt-4">{error}</p>}

      <div className="mt-10">
        <h2 className="text-2xl font-semibold mb-6">
          {joined ? "You're in the meeting" : 'Waiting to join'}
        </h2>

        {/* Local video */}
        <video
          ref={localVideoRef}
          autoPlay
          muted
          playsInline
          controls
          className="mx-auto w-80 aspect-video border-2 border-black rounded-lg shadow-md mb-6"
        />

        {/* Remote Videos */}
        <div className="flex flex-wrap justify-center gap-6">
          {remoteVideoStreams.map((remoteStream, idx) => (
            <video
              key={`video-${idx}`}
              autoPlay
              playsInline
              controls
              muted={false}
              className="w-80 aspect-video border-2 border-blue-600 rounded-lg shadow-lg"
              ref={(videoElement) => {
                if (videoElement && !videoElement.srcObject) {
                  videoElement.srcObject = remoteStream;
                }
              }}
            />
          ))}
        </div>

        {/* Remote Audios */}
        <div>
          {remoteAudioStreams.map((remoteStream, idx) => (
            <audio
              key={`audio-${idx}`}
              autoPlay
              controls
              muted={false}
              ref={(audioElement) => {
                if (audioElement && !audioElement.srcObject) {
                  audioElement.srcObject = remoteStream;
                }
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
