"""Smoke test v2: capture server stderr for debugging."""
import subprocess, time, sys, urllib.request, json, threading

proc = None
stderr_lines = []

def kill():
    global proc
    if proc and proc.poll() is None:
        proc.kill()
        proc.wait()

try:
    proc = subprocess.Popen(
        [sys.executable, 'serve_v04.py', '--provider', 'qwen', '--port', '8768'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=r'C:\Users\user\Desktop\voxline-ai-core',
    )

    def reader():
        for line in proc.stderr:
            stderr_lines.append(line.decode(errors='replace').rstrip())
    threading.Thread(target=reader, daemon=True).start()

    print('Loading Qwen...')
    for i in range(90):
        time.sleep(1)
        try:
            r = urllib.request.urlopen('http://127.0.0.1:8768/health', timeout=3)
            data = json.loads(r.read())
            print('Health:', data)
            break
        except:
            if i % 10 == 0:
                print(f'  waiting ({i}s)...')

    # Chat
    req = urllib.request.Request('http://127.0.0.1:8768/api/chat',
        data=json.dumps({'message': 'Hello'}).encode(),
        headers={'Content-Type': 'application/json'})
    r = urllib.request.urlopen(req, timeout=120)
    d = json.loads(r.read())
    print(f'Chat OK: {d["response"][:80]}')

    # Business
    req = urllib.request.Request('http://127.0.0.1:8768/api/business',
        data=json.dumps({'message': 'Analyze sales trends'}).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        r = urllib.request.urlopen(req, timeout=120)
        d = json.loads(r.read())
        print(f'Business OK: {d["response"][:80]}')
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f'Business FAILED ({e.code}): {body}')

    # Print server errors
    time.sleep(1)
    print('\n--- Server stderr (last 30 lines) ---')
    for line in stderr_lines[-30:]:
        print(line)

except Exception as e:
    print(f'FAIL: {e}')
    import traceback
    traceback.print_exc()
finally:
    kill()
    print('\nServer stopped.')
