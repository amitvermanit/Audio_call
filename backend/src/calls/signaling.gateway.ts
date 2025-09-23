import {
    SubscribeMessage,
    WebSocketGateway,
    OnGatewayInit,
} from '@nestjs/websockets';
import { Socket, Server } from 'socket.io';
import { MediasoupService } from './Mediasoup.service';
import {
    Producer,
    RtpCapabilities,
    WebRtcTransport,
    MediaKind,
    AppData,
    Consumer,
    PlainTransport
} from 'mediasoup/node/lib/types';
import { randomInt } from 'crypto';
import { getPort } from './port';

const rooms = new Map<
    string,
    {
        producers: Map<string, Producer>;
    }
>();

const userTransports = new Map<string, WebRtcTransport>();
const userProducers = new Map<string, Producer>();
const userConsumers = new Map<string, Consumer>();
const translationTransports = new Map<string, PlainTransport>();
const translationProducers = new Map<string, Producer>();
const translationConsumers = new Map<string, Consumer>();

let io: Server;


@WebSocketGateway({
    cors: {
        origin: '*',
    },
})
export class SignalingGateway implements OnGatewayInit {
    constructor(private readonly mediasoupService: MediasoupService) { }

    afterInit(server: Server) {
        io = server;
        this.mediasoupService.initMediasoup();
        console.log('🚀 Socket.IO Gateway ready');
    }

    handleConnection(socket: Socket) {
        console.log(`User connected ====== : ${socket.id}`);
    }

    handleDisconnect(socket: Socket) {
        console.log(`User disconnected: ${socket.id}`);
        this.closeTransports(socket.id);
        this.closeProducers(socket.id);
        this.closeConsumers(socket.id);
        this.closeTranslationTransports(socket.id);
        this.closeTranslationProducers(socket.id);
        this.closeTranslationConsumers(socket.id);
    }

    @SubscribeMessage('get-rtp-capabilities')
    handleGetRtp(socket: Socket) {
        console.log('Getting RTP capabilities');
        
        const rtpCapabilities = this.mediasoupService.getRtpCapabilities();
        socket.emit('rtp-capabilities', rtpCapabilities);
    }

    @SubscribeMessage('create-transport')
    async handleCreateTransport(
        socket: Socket,
        data: { direction: 'send' | 'recv' },
    ) {
        const { direction } = data;
        const { transport, params } =
            await this.mediasoupService.createWebRtcTransport();

        userTransports.set(`${socket.id}-${direction}`, transport);
        socket.emit(`transport-created-${direction}`, params);

        if (direction === 'send') {
            socket.on('connect-transport-send', async ({ dtlsParameters }) => {
                await transport.connect({ dtlsParameters });
                socket.emit('transport-connected-send');
            });

            socket.on('produce', async ({ kind, rtpParameters, roomCode, userId }) => {
                if (kind !== 'audio' && kind !== 'video') {
                    socket.emit('produce-error', `Invalid media kind: ${kind}`);
                    return;
                }
                const targetLang = 'eng'; // Default target language
                const srcLang = 'te'; // Default source language
                const gender = "female";


                // Remove all characters except a-z, A-Z, 0-9, underscore, and dash
                const cleanSocketId = socket.id.replace(/[^a-zA-Z0-9_-]/g, '');

                // Then form the sessionId
                const sessionId = `${cleanSocketId}`;

                const producer = await transport.produce({
                    kind: kind as MediaKind,
                    rtpParameters,
                });
                userProducers.set(`${socket.id}-${kind}`, producer);


                let ffmpegProducer: Producer<AppData> | null = null;
                let ffmpegVideoProducer: Producer<AppData> | null = null;
                const recvTransport = await this.mediasoupService.createPlainTransport("recv");
                const audioPlainTransport = await this.mediasoupService.createPlainTransport("send");
                if (kind === 'audio') {
                    const rtpPort = getPort();
                    // [Mediasoup -> FFmpeg]
                    translationTransports.set(`${socket.id}-send`, audioPlainTransport);
                    await audioPlainTransport.connect({
                        ip: '127.0.0.1',
                        port: rtpPort,
                    });

                    const consumer = await audioPlainTransport.consume({
                        producerId: producer.id,
                        rtpCapabilities: this.mediasoupService.getRtpCapabilities(),
                    });
                    translationConsumers.set(`${socket.id}-audio`, consumer);

                    // [FFmpeg -> Mediasoup]

                    translationTransports.set(`${socket.id}-recv`, recvTransport);
                    await recvTransport.connect({
                        ip: '127.0.0.1',    // FFmpeg sends audio to this IP
                        port: recvTransport.tuple.localPort,   // FFmpeg sends audio to this port
                    });

                    const codec = consumer.rtpParameters.codecs[0];
                    const payloadType = codec.payloadType;
                    const codecName = codec.mimeType.split('/')[1];
                    const clockRate = codec.clockRate;
                    const channels = codec.channels || 2;
                    const ssrc = randomInt(1, 0x7FFFFFFF);

                    const payload = {
                        producerId: producer.id,
                        rtpPort: rtpPort,
                        ip: audioPlainTransport.tuple.localIp,
                        codec: codecName,
                        clockRate,
                        channels,
                        payloadType,
                        ssrc,
                        outputPort: recvTransport.tuple.localPort,
                        targetLang,
                        sessionId,
                        enableVoiceClone: false,
                        userId: producer.id,
                        gender,
                        srcLang,

                    };

                    fetch("http://0.0.0.0:2004/translation/initiate", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify(payload)
                    })
                        .then(response => response.json())
                        .then(data => {
                            console.log("✅ Translation pipeline initiated:", data);
                        })
                        .catch(error => {
                            console.error("❌ Error initiating translation pipeline:", error);
                        });

                    // Consume the audio data from FFmpeg
                    ffmpegProducer = await recvTransport.produce({
                        kind: 'audio',
                        rtpParameters: {
                            codecs: [
                                {
                                    mimeType: 'audio/opus',
                                    payloadType,
                                    clockRate,
                                    channels
                                },
                            ],
                            encodings: [{ ssrc }]
                        },
                    });
                    translationProducers.set(`${socket.id}-audio`, ffmpegProducer);
                }

                socket.join(roomCode);

                if (!rooms.has(roomCode)) {
                    rooms.set(roomCode, { producers: new Map() });
                }

                if (kind === 'audio') {
                    if (ffmpegProducer != null) {
                        rooms.get(roomCode)!.producers.set(`${socket.id}-ffmpeg`, ffmpegProducer);
                        // socket.to(roomCode).emit('new-producer', {
                        //     producerId: ffmpegProducer.id,
                        //     socketId: socket.id,
                        //     kind,
                        // });
                        //for single user
                        socket.emit('new-producer', {
                            producerId: ffmpegProducer.id,
                            socketId: socket.id,
                            kind,
                        });
                        // socket.emit('new-producer', {
                        //     producerId: producer.id,
                        //     socketId: socket.id,
                        //     kind,
                        //     type : "original"
                        // });
                        // console.log("sent original")
                    }
                } else {
                    rooms.get(roomCode)!.producers.set(socket.id, producer);
                    socket.to(roomCode).emit('new-producer', {
                        producerId: producer.id,
                        socketId: socket.id,
                        kind,
                    });
                }


                // =======
                if (kind === 'video') {
                    const rtpPort = getPort();  // Allocate a dynamic RTP port
                    // const rtpPort = 25001;  // Allocate a dynamic RTP port
                    const videoPlainTransport = await this.mediasoupService.createPlainTransport("send");
                    translationTransports.set(`${socket.id}-send`, videoPlainTransport);
                    translationTransports.set(`${socket.id}-video`, videoPlainTransport);
                    await videoPlainTransport.connect({
                        ip: '127.0.0.1',
                        port: rtpPort,
                    });

                    const consumer = await videoPlainTransport.consume({
                        producerId: producer.id,
                        rtpCapabilities: this.mediasoupService.getRtpCapabilities(),
                    });
                    translationConsumers.set(`${socket.id}-video`, consumer);

                    const codec = consumer.rtpParameters.codecs[0];
                    const payloadType = codec.payloadType;
                    const codecName = codec.mimeType.split('/')[1]; // Should be "H264"
                    const clockRate = codec.clockRate;
                    const ssrc = randomInt(1, 0x7FFFFFFF);

                    const payload = {
                        rtpPort: rtpPort,
                        ip: videoPlainTransport.tuple.localIp,
                        codec: codecName,
                        clockRate,
                        payloadType,
                        targetPort: recvTransport.tuple.localPort,
                        sessionId,
                    };

                    fetch("http://0.0.0.0:2003/video/initiate", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload),
                    })
                        .then(res => res.json())
                        .then(data => console.log("✅ Video capture pipeline initiated:", data))
                        .catch(err => console.error("❌ Error initiating video capture pipeline:", err));

                    console.log(`🔄 Video Plain Transport created for session ${sessionId} with RTP port ${rtpPort}, ${recvTransport.tuple.localPort}`);
                    ffmpegVideoProducer = await recvTransport.produce({
                        kind: 'video',
                        rtpParameters: {
                            codecs: [
                                {
                                    mimeType: "video/VP8",
                                    payloadType: 100, // match Mediasoup router's preferredPayloadType
                                    clockRate: 90000,
                                    parameters: {
                                        "packetization-mode": 1,
                                        "profile-level-id": "42e01f",
                                        "level-asymmetry-allowed": 1
                                    },
                                    rtcpFeedback: [
                                        { type: "nack" },
                                        { type: "nack", parameter: "pli" },
                                        { type: "ccm", parameter: "fir" },
                                        { type: "goog-remb" }
                                    ]
                                }
                            ],
                            encodings: [{ ssrc: ssrc }]
                        },
                    })

                    // setInterval(() => {
                    //     ffmpegVideoProducer?.getStats().then(stats => {
                    //         console.log(`FFmpeg Video Producer Stats: ${JSON.stringify(stats)}`);
                    //     }).catch(err => {
                    //         console.error(`Error getting FFmpeg Video Producer stats: ${err}`);
                    //     });
                    // }, 1000); // Keep the connection alive

                    io.emit('new-producer', {
                        producerId: ffmpegVideoProducer.id,
                        socketId: socket.id,
                        kind: 'video',
                    });
                }
                // ====

                socket.emit('produced', { id: producer.id });
            });
        }

        if (direction === 'recv') {
            socket.on('connect-transport-recv', async ({ dtlsParameters }) => {
                await transport.connect({ dtlsParameters });
                socket.emit('transport-connected-recv');
            });
        }
    }

    @SubscribeMessage('consume')
    async handleConsume(
        socket: Socket,
        {
            producerId,
            rtpCapabilities,
        }: {
            producerId: string;
            rtpCapabilities: RtpCapabilities;
        },
    ) {
        const router = this.mediasoupService.getRouter();

        if (!router.canConsume({ producerId, rtpCapabilities })) {
            socket.emit('consume-error', 'Cannot consume this stream');
            return;
        }

        const transport = userTransports.get(`${socket.id}-recv`);
        if (!transport) {
            socket.emit('consume-error', 'No transport found');
            return;
        }

        const consumer = await transport.consume({
            producerId,
            rtpCapabilities,
            paused: false,
        });

        socket.emit('consumed', {
            id: consumer.id,
            kind: consumer.kind,
            rtpParameters: consumer.rtpParameters,
            producerId,
        });

        await consumer.resume();
    }


    closeTransports(socketId: string) {
        const sendTransport = userTransports.get(`${socketId}-send`);
        const recvTransport = userTransports.get(`${socketId}-recv`);
        if (sendTransport) {
            sendTransport.close();
            userTransports.delete(`${socketId}-send`);
        }
        if (recvTransport) {
            recvTransport.close();
            userTransports.delete(`${socketId}-recv`);
        }
    }

    closeProducers(socketId: string) {
        const audioProducer = userProducers.get(socketId + '-audio');
        if (audioProducer) {
            audioProducer.close();
            userProducers.delete(socketId + '-audio');
        }
        const videoProducer = userProducers.get(socketId + '-video');
        if (videoProducer) {
            videoProducer.close();
            userProducers.delete(socketId + '-video');
        }
    }

    closeConsumers(socketId: string) {
        const audioConsumer = userConsumers.get(socketId + '-audio');
        if (audioConsumer) {
            audioConsumer.close();
            userConsumers.delete(socketId + '-audio');
        }
        const videoConsumer = userConsumers.get(socketId + '-video');
        if (videoConsumer) {
            videoConsumer.close();
            userConsumers.delete(socketId + '-video');
        }
    }

    closeTranslationTransports(socketId: string) {
        const sendTransport = translationTransports.get(`${socketId}-send`);
        const recvTransport = translationTransports.get(`${socketId}-recv`);
        const videoTransport = translationTransports.get(`${socketId}-video`);

        if (sendTransport) {
            sendTransport.close();
            translationTransports.delete(`${socketId}-send`);
        }
        if (recvTransport) {
            recvTransport.close();
            translationTransports.delete(`${socketId}-recv`);
        }
        if (videoTransport) {
            videoTransport.close();
            translationTransports.delete(`${socketId}-video`);
        }
    }

    closeTranslationProducers(socketId: string) {
        const audioProducer = translationProducers.get(socketId + '-audio');
        if (audioProducer) {
            audioProducer.close();
            translationProducers.delete(socketId + '-audio');
        }
        const videoProducer = translationProducers.get(socketId + '-video');
        if (videoProducer) {
            videoProducer.close();
            translationProducers.delete(socketId + '-video');
        }
    }

    closeTranslationConsumers(socketId: string) {
        const audioConsumer = translationConsumers.get(socketId + '-audio');
        if (audioConsumer) {
            audioConsumer.close();
            translationConsumers.delete(socketId + '-audio');
        }
        const videoConsumer = translationConsumers.get(socketId + '-video');
        if (videoConsumer) {
            videoConsumer.close();
            translationConsumers.delete(socketId + '-video');
        }
    }

}
