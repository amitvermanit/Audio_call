import { Module } from '@nestjs/common';
import { SignalingGateway } from './signaling.gateway';
import { MediasoupService } from './Mediasoup.service';

@Module({
    providers: [SignalingGateway, MediasoupService],
})
export class SignalingModule { }
