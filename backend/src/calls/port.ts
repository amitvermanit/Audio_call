// code adapted from src/lib/port.ts creating a getPortPair fuinction as well

const MIN_PORT = 20000;
const MAX_PORT = 30000;

const takenPortSet = new Set<number>();

export const getPort = (): number => {
    let port = getRandomPort();

    while (takenPortSet.has(port)) {
        port = getRandomPort();
    }

    takenPortSet.add(port);
    return port;
};

export const getPortPair = (): [number, number] => {
    let port = getRandomPort();

    while (takenPortSet.has(port) || takenPortSet.has(port + 1)) {
        port = getRandomPort();
    }

    takenPortSet.add(port);
    takenPortSet.add(port+1);
    return [port, port+1];
};

export const releasePort = (port: number): void => {
    takenPortSet.delete(port);
};

const getRandomPort = (): number => {
    return Math.floor(Math.random() * (MAX_PORT - MIN_PORT + 1) + MIN_PORT);
};
