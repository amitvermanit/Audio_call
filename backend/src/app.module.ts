import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { SignalingModule } from './calls/signaling.module';

@Module({
  imports: [SignalingModule],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
