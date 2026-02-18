import os
import time
import subprocess
import glob
import json

API_URL = "http://localhost:3002"
PDF_DIR = "examples/pdfs"

def wait_for_server():
    print("Waiting for server to be ready...")
    retries = 30
    while retries > 0:
        try:
            result = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", f"{API_URL}/health"],
                capture_output=True,
                text=True
            )
            if result.stdout.strip() == "200":
                print("Server is ready!")
                return True
        except Exception:
            pass
        time.sleep(1)
        retries -= 1
    print("Server failed to start.")
    return False

def run_load_test():
    if not wait_for_server():
        return

    pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    print(f"Found {len(pdf_files)} PDFs to process.")

    for i, pdf_path in enumerate(pdf_files):
        filename = os.path.basename(pdf_path)
        print(f"[{i+1}/{len(pdf_files)}] Processing {filename}...")
        
        start_time = time.time()
        try:
            # Use curl for multipart upload
            # curl -F "file=@path/to/file" http://localhost:3002/parse
            result = subprocess.run(
                [
                    "curl", 
                    "-s", 
                    "-w", "\n%{http_code}", 
                    "-F", f"file=@{pdf_path}", 
                    f"{API_URL}/parse"
                ],
                capture_output=True,
                text=True
            )
            
            elapsed = time.time() - start_time
            stdout_lines = result.stdout.strip().split('\n')
            status_code = stdout_lines[-1] if stdout_lines else "000"
            
            if status_code == "200":
                print(f"  Success! Took {elapsed:.2f}s")
            else:
                print(f"  Failed! Status: {status_code}, Took {elapsed:.2f}s")
                # print(f"  Response: {result.stdout[:200]}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    run_load_test()
