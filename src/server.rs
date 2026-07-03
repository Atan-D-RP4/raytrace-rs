use crate::film::SharedFramebuffer;

use futures_lite::AsyncWriteExt;
use futures_lite::io::{AsyncRead, AsyncReadExt, AsyncWrite};
use smol::net::TcpListener;
use tracing::info;

pub struct Prepend<S: AsyncRead + AsyncWrite + Unpin> {
    inner: S,
    buf: Vec<u8>,
    pos: usize,
}

impl<S: AsyncRead + AsyncWrite + Unpin> Prepend<S> {
    pub fn new(inner: S, buf: Vec<u8>) -> Self {
        Self { inner, buf, pos: 0 }
    }
}

impl<S: AsyncRead + AsyncWrite + Unpin> AsyncRead for Prepend<S> {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut [u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        let this = self.get_mut();
        if this.pos < this.buf.len() {
            let n = std::cmp::min(buf.len(), this.buf.len() - this.pos);
            buf[..n].copy_from_slice(&this.buf[this.pos..this.pos + n]);
            this.pos += n;
            std::task::Poll::Ready(Ok(n))
        } else {
            let inner = std::pin::Pin::new(&mut this.inner);
            inner.poll_read(cx, buf)
        }
    }
}

impl<S: AsyncRead + AsyncWrite + Unpin> AsyncWrite for Prepend<S> {
    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        let this = self.get_mut();
        let inner = std::pin::Pin::new(&mut this.inner);
        inner.poll_write(cx, buf)
    }

    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let this = self.get_mut();
        let inner = std::pin::Pin::new(&mut this.inner);
        inner.poll_flush(cx)
    }

    fn poll_close(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let this = self.get_mut();
        let inner = std::pin::Pin::new(&mut this.inner);
        inner.poll_close(cx)
    }
}

pub fn html_client() -> String {
    r#"<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Raytracer</title></head>
<body style="margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh">
<canvas id="c"></canvas>
<script>
const ws = new WebSocket(`ws://${location.host}/ws`);
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
ws.binaryType = 'arraybuffer';
let npass = 0;
ws.onmessage = (e) => {
  const buf = new Uint8Array(e.data);
  const view = new DataView(buf.buffer);
  const w = view.getUint32(0, true);
  const h = view.getUint32(4, true);
  canvas.width = w; canvas.height = h;
  const img = ctx.createImageData(w, h);
  for (let i = 8, j = 0; i < buf.length; i += 3, j += 4) {
    img.data[j] = buf[i]; img.data[j+1] = buf[i+1];
    img.data[j+2] = buf[i+2]; img.data[j+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  document.title = `Raytracer #${++npass}`;
};
</script>
</body>
</html>"#
    .to_string()
}

pub async fn run(framebuffer: SharedFramebuffer) -> std::io::Result<()> {
    let addr = "0.0.0.0:3000";
    let listener = TcpListener::bind(addr).await?;
    info!("Server listening on http://{}", addr);
    loop {
        let (mut stream, addr) = listener.accept().await?;
        println!("Accepted connection from {}", addr);
        let framebuffer = framebuffer.clone();
        smol::spawn(async move {
            if let Err(e) = handle_connection(&mut stream, framebuffer).await {
                eprintln!("Connection error: {}", e);
            }
        })
        .detach();
    }
}

pub async fn handle_connection<S: AsyncRead + AsyncWrite + Unpin>(
    stream: &mut S,
    framebuffer: SharedFramebuffer,
) -> std::io::Result<()> {
    let mut buf = [0u8; 4096];
    let n = stream.read(&mut buf).await?;
    if n == 0 {
        return Ok(());
    }

    let request = String::from_utf8_lossy(&buf[..n]);
    if request.starts_with("GET /ws") {
        let mut prepend = Prepend::new(stream, buf[..n].to_vec());
        handle_websocket(&mut prepend, framebuffer).await?;
    } else {
        handle_http(stream).await?;
    }
    Ok(())
}

async fn handle_http<S: AsyncWrite + Unpin>(stream: &mut S) -> std::io::Result<()> {
    let html = html_client();
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        html.len(),
        html
    );
    stream.write_all(response.as_bytes()).await?;
    Ok(())
}

async fn handle_websocket<S: AsyncRead + AsyncWrite + Unpin>(
    stream: &mut S,
    framebuffer: SharedFramebuffer,
) -> std::io::Result<()> {
    // Perform WebSocket handshake
    let mut ws = async_tungstenite::accept_async(stream).await.map_err(|e| {
        eprintln!("WebSocket handshake error: {}", e);
        std::io::Error::other("WebSocket handshake error")
    })?;

    // Now we can send framebuffer updates over the WebSocket
    loop {
        let message = {
            let fb = framebuffer.read().map_err(|e| {
                eprintln!("Framebuffer lock error: {}", e);
                std::io::Error::other("Framebuffer lock error")
            })?;
            let (w, h) = fb.image.dimensions();
            let raw = fb.image.as_raw();
            let mut message = Vec::with_capacity(8 + raw.len());
            message.extend_from_slice(&w.to_le_bytes());
            message.extend_from_slice(&h.to_le_bytes());
            message.extend_from_slice(raw);
            message
        };

        ws.send(async_tungstenite::tungstenite::Message::Binary(
            message.into(),
        ))
        .await
        .map_err(|e| {
            eprintln!("WebSocket send error: {}", e);
            std::io::Error::other("WebSocket send error")
        })?;

        // Wait for a short duration before sending the next update.
        // Basically an FPS limiter to avoid overwhelming the client with updates.
        smol::Timer::after(std::time::Duration::from_millis(1000)).await;
    }
}
