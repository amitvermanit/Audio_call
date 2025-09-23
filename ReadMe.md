##Backend:
cd backend
open constants.ts file, replace announced ip with your ipv4 address
run command: npm run start:dev

##Frontend:
cd frontend
run command: npm run dev

##python:
cd ml
run command: python webrtc_python.py

instructions:
run backend before you run python server

To reproduce:

1. Start the backend and frotend servers
2. Run the webrtc_python.py file
3. Open http://localhost:5173 on chrome, and open another instance of localhost:5173 on incognito in chrome
4. Click join on both browsers
5. The audio from the second run instance is sent to the python server.
6. You can here the processed audio on the first run instance with some delay or slow motion.
7. For debugging the input streams you can use "save_wav_from_bytes" in webtrc_python.py file and pass necessary parameters to it.
