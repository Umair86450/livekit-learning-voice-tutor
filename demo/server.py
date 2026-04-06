from http.server import HTTPServer, BaseHTTPRequestHandler
from livekit.api import AccessToken, VideoGrants
import os, json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
ROOM = "demo-room"

def _ws_url_for_request(handler: BaseHTTPRequestHandler) -> str:
    """Use client's host so same-network demos work (client opens http://YOUR_IP:8080 -> ws://YOUR_IP:7880)."""
    host_header = handler.headers.get("Host", "").strip()
    if host_header and "localhost" not in host_header and "127.0.0.1" not in host_header:
        host = host_header.split(":")[0]
        return f"ws://{host}:7880"
    return LIVEKIT_URL


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/token":
            ws_url = _ws_url_for_request(self)
            token = AccessToken(API_KEY, API_SECRET)
            token.with_identity("browser-user")
            token.with_grants(VideoGrants(room_join=True, room=ROOM))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "token": token.to_jwt(),
                "url": ws_url
            }).encode())
        else:
            html_path = os.path.join(os.path.dirname(__file__), "index.html")
            with open(html_path, "rb") as f:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(f.read())

    def log_message(self, *args): pass

HTTPServer(("", 8080), Handler).serve_forever()
