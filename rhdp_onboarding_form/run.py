from app import app
import socket

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

if __name__ == '__main__':
    port = find_free_port()
    print(f"Starting server on port {port}")
    app.run(debug=True, port=port)
