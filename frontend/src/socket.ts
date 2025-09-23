import { io } from 'socket.io-client';

const socketUrl = 'http://localhost:3000';
export const socket = io(socketUrl, {
  transports: ['websocket'],
});
