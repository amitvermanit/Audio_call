// src/mediasoup/mediasoup.service.ts
import * as mediasoup from 'mediasoup';
import { Injectable } from '@nestjs/common';
import { Router, RtpCapabilities, WebRtcTransport } from 'mediasoup/node/lib/types';
import { Worker } from 'mediasoup/node/lib/types';
import { msConfig } from './constants';

@Injectable()
export class MediasoupService {
    private worker: Worker;
    private router: Router;

    async initMediasoup() {
        this.worker = await mediasoup.createWorker({
            logLevel: 'debug',
            rtcMinPort: 20000,
            rtcMaxPort: 49999,
        });

        this.router = await this.worker.createRouter({
            mediaCodecs: [
                {
                    kind: 'audio',
                    mimeType: 'audio/opus',
                    clockRate: 48000,
                    channels: 2,
                },
                {
                    kind: 'video',
                    mimeType: 'video/VP8',
                    clockRate: 90000,
                },
            ],
        });

        console.log('✅ MediaSoup initialized');
    }

    getRtpCapabilities(): RtpCapabilities {
        return this.router.rtpCapabilities;
    }

    getRouter(): Router {
        return this.router;
    }

    async createWebRtcTransport(): Promise<{
        transport: WebRtcTransport,
        params: {
            id: string,
            iceParameters: any,
            iceCandidates: any[],
            dtlsParameters: any,
        },
    }> {
        const transport = await this.router.createWebRtcTransport({
            listenIps: [{ ip: '0.0.0.0', announcedIp: msConfig.announcedIp }],
            enableUdp: true,
            enableTcp: true,
            preferUdp: true,
        });

        return {
            transport,
            params: {
                id: transport.id,
                iceParameters: transport.iceParameters,
                iceCandidates: transport.iceCandidates,
                dtlsParameters: transport.dtlsParameters,
            },
        };
    }


    async createPlainTransport(type: 'send' | 'recv') {
        const plainTransport = await this.router.createPlainTransport({
            listenIp: { ip: '0.0.0.0', announcedIp: msConfig.announcedIp }, // for external access
            rtcpMux: true,
            comedia: type === 'send' ? false : true,  // allow remote to connect first
        });

        return plainTransport;

    }
}
