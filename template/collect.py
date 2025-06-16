import time
import json
import os
import signal
import sys
import random
import traceback
import socket
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import database
from database import Database

WEBSITES = [
    # websites of your choice
    "https://cse.buet.ac.bd/moodle/",
    "https://google.com",
    "https://prothomalo.com",
]

TRACES_PER_SITE = 10
FINGERPRINTING_URL = "http://localhost:5001" 
OUTPUT_PATH = "dataset.json"

# Initialize the database to save trace data reliably
database.db = Database(WEBSITES)

""" Signal handler to ensure data is saved before quitting. """
def signal_handler(sig, frame):
    print("\nReceived termination signal. Exiting gracefully...")
    try:
        database.db.export_to_json(OUTPUT_PATH)
    except:
        pass
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


"""
Some helper functions to make your life easier.
"""

def is_server_running(host='127.0.0.1', port=5001):
    """Check if the Flask server is running."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
    except (socket.error, ConnectionRefusedError):
        return False
    finally:
        sock.close()
    return True

def setup_webdriver():
    """Set up the Selenium WebDriver with Chrome options."""
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    # Let Selenium manage the driver automatically
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def retrieve_traces_from_backend(driver):
    """Retrieve traces from the backend API."""
    traces = driver.execute_script("""
        return fetch('/api/get_results')
            .then(response => response.ok ? response.json() : {traces: []})
            .then(data => data.traces || [])
            .catch(() => []);
    """)
    
    count = len(traces) if traces else 0
    print(f"  - Retrieved {count} traces from backend API" if count else "  - No traces found in backend storage")
    return traces or []

def clear_trace_results(driver, wait):
    """Clear all results from the backend by pressing the button."""
    clear_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Clear all results')]")
    clear_button.click()

    wait.until(EC.text_to_be_present_in_element(
        (By.XPATH, "//div[@role='alert']"), "Cleared"))
    
def is_collection_complete():
    """Check if target number of traces have been collected."""
    current_counts = database.db.get_traces_collected()
    remaining_counts = {website: max(0, TRACES_PER_SITE - count) 
                      for website, count in current_counts.items()}
    return sum(remaining_counts.values()) == 0

"""
Your implementation starts here.
"""

def collect_single_trace(driver, wait, website_url):
    """ Implement the trace collection logic here. 
    1. Open the fingerprinting website
    2. Click the button to collect trace
    3. Open the target website in a new tab
    4. Interact with the target website (scroll, click, etc.)
    5. Return to the fingerprinting tab and close the target website tab
    6. Wait for the trace to be collected
    7. Return success or failure status
    """

def collect_fingerprints(driver):
    """
    Main logic to collect fingerprints.
    - Loop until the target number of traces for each website is collected.
    - For each website, open it in a new tab, then switch back to the
      fingerprinting tab to start the collection.
    - Wait for the collection to complete before moving to the next one.
    """
    wait = WebDriverWait(driver, 20)  # 20-second timeout

    # Open the fingerprinting page
    driver.get(FINGERPRINTING_URL)
    print(f"Opened fingerprinting page: {FINGERPRINTING_URL}")
    
    try:
        print("  - Waiting for application to initialize...")
        wait.until(
            lambda d: d.execute_script("return window.Alpine && window.Alpine.store && window.Alpine.store('app')")
        )
        print("  - Application is ready.")
    except Exception:
        print("  - Fatal: Timed out waiting for Alpine.js to initialize on the page.")
        print("  - Please check that the server is running and the page is loading correctly.")
        return

    # Main collection loop
    while not is_collection_complete():
        current_counts = database.db.get_traces_collected()
        print("\n--- Current Trace Counts ---")
        for site, count in current_counts.items():
            print(f"- {site}: {count}/{TRACES_PER_SITE}")
        print("--------------------------")

        # Find a site that needs more traces
        for i, site_url in enumerate(WEBSITES):
            if current_counts.get(site_url, 0) < TRACES_PER_SITE:
                target_site = site_url
                target_idx = i
                break
        else:
            # Should not happen if is_collection_complete is correct
            print("Collection seems complete, but loop continued. Exiting.")
            break

        print(f"\nCollecting new trace for: {target_site}")

        # Open target website in a new tab
        driver.switch_to.new_window('tab')
        try:
            driver.get(target_site)
            print(f"  - Opened target site: {target_site}")
            # Wait for site to be interactive and maybe scroll
            time.sleep(random.uniform(5, 8))
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(random.uniform(2, 4))
        except Exception as e:
            print(f"  - Error opening or interacting with {target_site}: {e}")
            if len(driver.window_handles) > 1:
                driver.close() # Close the failed tab
            driver.switch_to.window(driver.window_handles[0])
            time.sleep(1)
            continue # Skip to next attempt

        # Close the target website tab before starting collection
        if len(driver.window_handles) > 1:
            driver.switch_to.window(driver.window_handles[1])
            driver.close()
        
        # Switch back to the main fingerprinting tab
        driver.switch_to.window(driver.window_handles[0])
        time.sleep(1)

        # Execute collection script
        print("  - Starting trace collection...")
        driver.execute_script(f"window.Alpine.store('app').collectTraceData('{target_site}', {target_idx})")

        # Wait for collection to finish by polling the status
        wait.until(EC.text_to_be_present_in_element(
            (By.XPATH, "//div[@role='alert']"), "Successfully saved trace"))
        print("  - Trace collected and saved successfully.")
        time.sleep(2) # Brief pause before next collection

def main():
    if not is_server_running(port=5001):
        print("Error: Flask server is not running on port 5001.")
        print("Please start the server in a separate terminal with `python app.py` and try again.")
        sys.exit(1)

    database.db.init_database()

    driver = None
    try:
        driver = setup_webdriver()
        collect_fingerprints(driver)
    except Exception as e:
        print(f"\nAn error occurred during collection: {traceback.format_exc()}")
    finally:
        if driver:
            driver.quit()
            print("\nBrowser closed.")

    print("\n--- Final Trace Counts ---")
    final_counts = database.db.get_traces_collected()
    for site, count in final_counts.items():
        print(f"- {site}: {count}/{TRACES_PER_SITE}")
    
    print("\nExporting final dataset to JSON...")
    database.db.export_to_json(OUTPUT_PATH)
    
    if not is_collection_complete():
        print("\nWarning: Collection did not complete. Run the script again to collect remaining traces.")
    else:
        print("\nCollection complete!")

if __name__ == "__main__":
    main()
