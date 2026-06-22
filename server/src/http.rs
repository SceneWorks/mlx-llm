//! A minimal HTTP/1.1 request reader — just enough to serve a single JSON/SSE endpoint without an
//! HTTP-framework dependency. Reads the request line, headers, and a `Content-Length` body; the
//! server replies `Connection: close`, so one request per connection (no keep-alive/chunked-body
//! handling, which a real gateway would add).

use std::io::{self, BufRead};

/// Reject absurd request bodies up front (a hostile or buggy `Content-Length`).
const MAX_BODY: usize = 16 * 1024 * 1024;

/// A parsed request: method, path (query stripped), and the raw body bytes.
#[derive(Debug, PartialEq, Eq)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

/// Read one request from `reader`. `Ok(None)` means the peer closed before sending anything (a clean
/// idle disconnect). A malformed request line or an over-cap body is an [`io::Error`].
pub fn read_request(reader: &mut impl BufRead) -> io::Result<Option<Request>> {
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None); // peer closed with no request
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let target = parts.next().unwrap_or_default().to_string();
    if method.is_empty() || target.is_empty() {
        return Err(invalid("malformed request line"));
    }
    // Strip any query string for routing.
    let path = target.split('?').next().unwrap_or("").to_string();

    // Headers until a blank line; we only need Content-Length.
    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 {
            return Err(invalid("unexpected EOF in headers"));
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((name, value)) = h.split_once(':') {
            if name.trim().eq_ignore_ascii_case("content-length") {
                content_length = value
                    .trim()
                    .parse()
                    .map_err(|_| invalid("bad Content-Length"))?;
            }
        }
    }
    if content_length > MAX_BODY {
        return Err(invalid("request body too large"));
    }

    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;
    Ok(Some(Request { method, path, body }))
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parses_post_with_body() {
        let raw = "POST /v1/chat/completions?x=1 HTTP/1.1\r\n\
                   Host: localhost\r\n\
                   Content-Type: application/json\r\n\
                   Content-Length: 14\r\n\
                   \r\n\
                   {\"hello\":\"hi\"}";
        let mut c = Cursor::new(raw.as_bytes());
        let req = read_request(&mut c).unwrap().unwrap();
        assert_eq!(req.method, "POST");
        assert_eq!(req.path, "/v1/chat/completions"); // query stripped
        assert_eq!(req.body, b"{\"hello\":\"hi\"}");
    }

    #[test]
    fn parses_get_without_body() {
        let mut c = Cursor::new(&b"GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n"[..]);
        let req = read_request(&mut c).unwrap().unwrap();
        assert_eq!(req.method, "GET");
        assert_eq!(req.path, "/v1/models");
        assert!(req.body.is_empty());
    }

    #[test]
    fn content_length_case_insensitive() {
        let mut c = Cursor::new(&b"POST / HTTP/1.1\r\ncontent-length: 2\r\n\r\nhi"[..]);
        let req = read_request(&mut c).unwrap().unwrap();
        assert_eq!(req.body, b"hi");
    }

    #[test]
    fn empty_stream_is_none() {
        let mut c = Cursor::new(&b""[..]);
        assert_eq!(read_request(&mut c).unwrap(), None);
    }

    #[test]
    fn oversized_body_rejected() {
        let raw = format!("POST / HTTP/1.1\r\nContent-Length: {}\r\n\r\n", MAX_BODY + 1);
        let mut c = Cursor::new(raw.into_bytes());
        assert!(read_request(&mut c).is_err());
    }
}
