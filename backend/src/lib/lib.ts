import { MediaKind, RtpParameters } from "mediasoup/node/lib/rtpParametersTypes";

export const getCodecInfoFromRtpParameters = (kind: MediaKind, rtpParameters: RtpParameters) => {
    return {
        payloadType: rtpParameters.codecs[0].payloadType,
        codecName: rtpParameters.codecs[0].mimeType.replace(`${kind}/`, ''),
        clockRate: rtpParameters.codecs[0].clockRate,
        channels: kind === 'audio' ? rtpParameters.codecs[0].channels : undefined
    };
};