use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::{mpsc, watch};

use crate::multiplexer::{IncomingMessage, MultiplexerError, StreamParser};

const READ_BUF_SIZE: usize = 64 * 1024;
/// Capacity of the bounded frame channel — provides backpressure for low-priority traffic.
/// Must be small to prevent frame accumulation: at 60 fps each extra slot adds ~16 ms
/// of end-to-end latency.  A depth of 2 allows one frame in-flight (being written to
/// the pipe) while the next one is queued.
const FRAME_QUEUE_DEPTH: usize = 2;
/// Capacity of the incoming message channel. Must be small to prevent frame
/// accumulation: at 60 fps, each extra slot adds ~16 ms of display latency.
/// A depth of 2 allows one frame in-flight while another is being processed.
const INCOMING_QUEUE_DEPTH: usize = 2;

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("multiplexer error: {0}")]
    Multiplexer(#[from] MultiplexerError),
    #[error("transport disconnected")]
    Disconnected,
}

/// Outgoing sender handle, cheaply cloneable.
///
/// Three priority levels:
/// - `send_input`  → input events: unbounded, never blocked, always sent first.
/// - `send_anchor` → anchor frames: unbounded, never blocked, sent second.
/// - `try_send_frame` → regular frame chunks: bounded (`FRAME_QUEUE_DEPTH`),
///   returns `false` under backpressure so the caller can drop the chunk.
#[derive(Clone)]
pub struct TransportSender {
    input_tx: mpsc::UnboundedSender<Vec<u8>>,
    anchor_tx: mpsc::UnboundedSender<Vec<u8>>,
    frame_tx: mpsc::Sender<Vec<u8>>,
}

impl TransportSender {
    /// Send an input event. Always succeeds unless the transport is shut down.
    pub fn send_input(&self, data: Vec<u8>) -> bool {
        self.input_tx.send(data).is_ok()
    }

    /// Send an anchor frame. Always succeeds unless the transport is shut down.
    pub fn send_anchor(&self, data: Vec<u8>) -> bool {
        self.anchor_tx.send(data).is_ok()
    }

    /// Enqueue a regular frame chunk. Returns `false` if the queue is full (caller drops it).
    pub fn try_send_frame(&self, data: Vec<u8>) -> bool {
        self.frame_tx.try_send(data).is_ok()
    }

    /// Number of frame slots currently occupied.
    pub fn frame_queue_len(&self) -> usize {
        self.frame_tx.max_capacity() - self.frame_tx.capacity()
    }
}

/// SSH transport: reads from a byte source, writes to a byte sink.
///
/// Spawn with [`SshTransport::run`]. The caller retains a [`TransportSender`]
/// for sending and a `Receiver<IncomingMessage>` for receiving.
///
/// # Note on I/O backend
/// Currently uses tokio's standard async I/O. Production hot path should use
/// `tokio-uring` for `io_uring`-backed zero-copy reads/writes on Linux.
pub struct SshTransport {
    sender: TransportSender,
    incoming_rx: mpsc::Receiver<IncomingMessage>,
    disconnect_rx: watch::Receiver<bool>,
}

impl SshTransport {
    /// Create a transport that wraps the given reader/writer pair and start the I/O loops.
    ///
    /// Returns `(SshTransport, task_handle)`. Drop `SshTransport` or close the underlying
    /// I/O to shut down.
    pub fn new<R, W>(reader: R, writer: W) -> (Self, tokio::task::JoinHandle<()>)
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        let (input_tx, input_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let (anchor_tx, anchor_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let (frame_tx, frame_rx) = mpsc::channel::<Vec<u8>>(FRAME_QUEUE_DEPTH);
        let (incoming_tx, incoming_rx) = mpsc::channel::<IncomingMessage>(INCOMING_QUEUE_DEPTH);
        let (disconnect_tx, disconnect_rx) = watch::channel(false);

        let sender = TransportSender {
            input_tx,
            anchor_tx,
            frame_tx,
        };

        let dtx_write = disconnect_tx.clone();
        let task = tokio::spawn(async move {
            tokio::select! {
                _ = read_loop(reader, incoming_tx, disconnect_tx) => {}
                _ = write_loop(writer, input_rx, anchor_rx, frame_rx, dtx_write) => {}
            }
        });

        let transport = Self {
            sender,
            incoming_rx,
            disconnect_rx,
        };

        (transport, task)
    }

    /// A cloneable sender handle for outgoing messages.
    pub fn sender(&self) -> TransportSender {
        self.sender.clone()
    }

    /// Receive the next incoming message. Returns `None` if transport closed.
    pub async fn recv(&mut self) -> Option<IncomingMessage> {
        self.incoming_rx.recv().await
    }

    /// Watch channel that becomes `true` when the connection is lost.
    pub fn disconnect_watch(&self) -> watch::Receiver<bool> {
        self.disconnect_rx.clone()
    }
}

async fn read_loop<R: AsyncRead + Unpin>(
    mut reader: R,
    msg_tx: mpsc::Sender<IncomingMessage>,
    disconnect_tx: watch::Sender<bool>,
) {
    let mut parser = StreamParser::new();
    let mut buf = vec![0u8; READ_BUF_SIZE];
    loop {
        match reader.read(&mut buf).await {
            Ok(0) => break,
            Ok(n) => match parser.push(&buf[..n]) {
                Ok(msgs) => {
                    for msg in msgs {
                        if msg_tx.send(msg).await.is_err() {
                            return;
                        }
                    }
                }
                Err(_) => break,
            },
            Err(_) => break,
        }
    }
    let _ = disconnect_tx.send(true);
}

async fn write_loop<W: AsyncWrite + Unpin>(
    mut writer: W,
    mut input_rx: mpsc::UnboundedReceiver<Vec<u8>>,
    mut anchor_rx: mpsc::UnboundedReceiver<Vec<u8>>,
    mut frame_rx: mpsc::Receiver<Vec<u8>>,
    disconnect_tx: watch::Sender<bool>,
) {
    loop {
        // biased: input > anchor > frame (priority order).
        let data = tokio::select! {
            biased;
            Some(d) = input_rx.recv() => d,
            Some(d) = anchor_rx.recv() => d,
            Some(d) = frame_rx.recv() => d,
            else => break,
        };
        if writer.write_all(&data).await.is_err() {
            break;
        }
        // Flush immediately so the data reaches the pipe/socket without
        // lingering in tokio's Blocking adapter buffer.
        if writer.flush().await.is_err() {
            break;
        }
    }
    let _ = disconnect_tx.send(true);
}

#[cfg(test)]
mod tests {
    use remoteway_proto::header::{FrameHeader, MsgType, flags};
    use zerocopy::IntoBytes;

    use super::*;

    fn make_frame_bytes(stream_id: u16, msg_type: MsgType, payload: &[u8]) -> Vec<u8> {
        let hdr = FrameHeader::new(
            stream_id,
            msg_type,
            flags::LAST_CHUNK,
            payload.len() as u32,
            0,
        );
        let mut out = Vec::new();
        out.extend_from_slice(hdr.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[tokio::test]
    async fn recv_single_message() {
        let frame = make_frame_bytes(1, MsgType::FrameUpdate, b"hello");
        let cursor = std::io::Cursor::new(frame);
        let (mut transport, _task) = SshTransport::new(cursor, tokio::io::sink());
        let msg = transport.recv().await.unwrap();
        assert_eq!(msg.payload, b"hello");
    }

    #[tokio::test]
    async fn send_and_recv_roundtrip() {
        let (client_rd, server_wr) = tokio::io::duplex(65536);
        let (server_rd, client_wr) = tokio::io::duplex(65536);

        let (client, _ct) = SshTransport::new(client_rd, client_wr);
        let (mut server, _st) = SshTransport::new(server_rd, server_wr);

        let data = make_frame_bytes(1, MsgType::FrameUpdate, b"ping");
        client.sender().send_anchor(data);

        let msg = server.recv().await.unwrap();
        assert_eq!(msg.payload, b"ping");
    }

    #[tokio::test]
    async fn input_bypasses_full_frame_queue() {
        // Keep receivers alive so channels are not considered closed.
        let (input_tx, _input_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let (anchor_tx, _anchor_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let (frame_tx, _frame_rx) = mpsc::channel::<Vec<u8>>(1);

        let sender = TransportSender {
            input_tx,
            anchor_tx,
            frame_tx,
        };

        // Fill the single-slot frame queue.
        assert!(sender.try_send_frame(vec![0u8; 16]));
        // Second frame is dropped — queue full.
        assert!(!sender.try_send_frame(vec![0u8; 16]));
        // Input always succeeds even when frame queue is full.
        assert!(sender.send_input(vec![0u8; 16]));
    }

    #[tokio::test]
    async fn disconnect_watch_fires_on_eof() {
        // Empty reader → immediate EOF.
        let empty: &[u8] = &[];
        let cursor = std::io::Cursor::new(empty);
        let (transport, _task) = SshTransport::new(cursor, tokio::io::sink());
        let mut watch = transport.disconnect_watch();

        // Wait until disconnected.
        watch
            .wait_for(|&v| v)
            .await
            .expect("watch closed unexpectedly");
        assert!(*watch.borrow());
    }

    #[tokio::test]
    async fn multiple_messages_in_order() {
        let mut all = Vec::new();
        for i in 0u8..5 {
            all.extend(make_frame_bytes(1, MsgType::FrameUpdate, &[i]));
        }
        let (mut transport, _task) =
            SshTransport::new(std::io::Cursor::new(all), tokio::io::sink());
        for i in 0u8..5 {
            let msg = transport.recv().await.unwrap();
            assert_eq!(msg.payload, &[i]);
        }
    }

    #[tokio::test]
    async fn large_payload_round_trip() {
        let (client_rd, server_wr) = tokio::io::duplex(512 * 1024);
        let (server_rd, client_wr) = tokio::io::duplex(512 * 1024);

        let (client, _ct) = SshTransport::new(client_rd, client_wr);
        let (mut server, _st) = SshTransport::new(server_rd, server_wr);

        let payload = vec![0xAB_u8; 256 * 1024];
        let data = make_frame_bytes(1, MsgType::FrameUpdate, &payload);
        client.sender().send_anchor(data);

        let msg = server.recv().await.unwrap();
        assert_eq!(msg.payload.len(), 256 * 1024);
        assert!(msg.payload.iter().all(|&b| b == 0xAB));
    }

    #[tokio::test]
    async fn concurrent_senders_three_clones() {
        let (client_rd, server_wr) = tokio::io::duplex(65536);
        let (server_rd, client_wr) = tokio::io::duplex(65536);

        let (client, _ct) = SshTransport::new(client_rd, client_wr);
        let (mut server, _st) = SshTransport::new(server_rd, server_wr);

        for i in 0u8..3 {
            let sender = client.sender();
            let data = make_frame_bytes(i as u16, MsgType::FrameUpdate, &[i]);
            tokio::spawn(async move {
                sender.send_anchor(data);
            });
        }

        let mut received = Vec::new();
        for _ in 0..3 {
            let msg = server.recv().await.unwrap();
            received.push(msg.payload[0]);
        }
        received.sort();
        assert_eq!(received, vec![0, 1, 2]);
    }

    #[tokio::test]
    async fn frame_queue_len_after_send() {
        let (input_tx, _input_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let (anchor_tx, _anchor_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let (frame_tx, _frame_rx) = mpsc::channel::<Vec<u8>>(4);

        let sender = TransportSender {
            input_tx,
            anchor_tx,
            frame_tx,
        };

        assert_eq!(sender.frame_queue_len(), 0);
        sender.try_send_frame(vec![1]);
        assert_eq!(sender.frame_queue_len(), 1);
        sender.try_send_frame(vec![2]);
        assert_eq!(sender.frame_queue_len(), 2);
    }
}
