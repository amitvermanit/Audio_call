##Backend:
cd backend
open constants.ts file, replace announced ip with your ipv4 address
run command: npm run start:dev

##Frontend:
cd frontend
run command: npm run dev

##python:
cd ml2
run command: python _main.py
cd pipelines/seamless_t2s
run command:python seamless_t2S_worker.py
cd ../..
cd pipelines/voice_clone
python voice_clone_worker.py


instructions:
run backend before you run python server

To reproduce:

1. Start the backend and frotend servers
2. Run the webrtc_python.py file
3. Open http://localhost:5173 on chrome, and open another instance of localhost:5173 on incognito in chrome
4. Click join on both browsers

