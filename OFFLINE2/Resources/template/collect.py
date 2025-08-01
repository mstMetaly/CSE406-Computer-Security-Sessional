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
from selenium.common.exceptions import TimeoutException
import database
from database import Database

# Custom expected condition to check for substring in element text
class text_to_be_present_in_element_value(object):
    def __init__(self, locator, text_):
        self.locator = locator
        self.text = text_

    def __call__(self, driver):
        try:
            element_text = driver.find_element(*self.locator).text
            return self.text in element_text
        except:
            return False

WEBSITES = [
    # websites of your choice
    "https://www.thedailystar.net/",
    "https://cse.buet.ac.bd/moodle/",
    "https://google.com",
    "https://prothomalo.com",
    
]

TRACES_PER_SITE = 500
FINGERPRINTING_URL = "http://localhost:5000" 
OUTPUT_PATH = "dataset.json"

# Scrolling configuration
SCROLL_PAUSE_TIME = 1.0
INTERACTION_TIME = 10
NUM_SCROLL_STEPS = 10

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

def is_server_running(host='127.0.0.1', port=5000):
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

def smooth_scroll_page(driver):
    """Perform smooth scrolling through the page."""
    try:
        # Get page height
        total_height = driver.execute_script("return document.body.scrollHeight")
        viewport_height = driver.execute_script("return window.innerHeight")
        current_position = 0
        
        # Scroll down smoothly
        while current_position < total_height:
            # Calculate next position with some randomness
            step = viewport_height / NUM_SCROLL_STEPS
            for i in range(NUM_SCROLL_STEPS):
                next_position = min(current_position + step, total_height)
                # Add some random jitter to make it more human-like
                jitter = random.uniform(-10, 10)
                scroll_script = f"window.scrollTo({{top: {next_position + jitter}, behavior: 'smooth'}})"
                driver.execute_script(scroll_script)
                time.sleep(SCROLL_PAUSE_TIME / NUM_SCROLL_STEPS)
                current_position = next_position
            
            # Random pause at each viewport
            time.sleep(random.uniform(0.5, 1.5))
        
        # Scroll back up smoothly
        while current_position > 0:
            step = viewport_height / NUM_SCROLL_STEPS
            for i in range(NUM_SCROLL_STEPS):
                next_position = max(current_position - step, 0)
                jitter = random.uniform(-10, 10)
                scroll_script = f"window.scrollTo({{top: {next_position + jitter}, behavior: 'smooth'}})"
                driver.execute_script(scroll_script)
                time.sleep(SCROLL_PAUSE_TIME / NUM_SCROLL_STEPS)
                current_position = next_position
            
            # Random pause at each viewport
            time.sleep(random.uniform(0.5, 1.5))
            
    except Exception as e:
        print(f"  - Error during scrolling: {e}")

def interact_with_page(driver):
    """Perform various interactions with the page."""
    try:
        # Try to find and click some interactive elements
        clickable_elements = driver.find_elements(By.CSS_SELECTOR, 
            'a, button, [role="button"], [type="button"], [type="submit"]')
        
        if clickable_elements:
            # Select a random subset of elements to interact with
            sample_size = min(3, len(clickable_elements))
            selected_elements = random.sample(clickable_elements, sample_size)
            
            for element in selected_elements:
                try:
                    # Scroll element into view smoothly
                    driver.execute_script(
                        "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", 
                        element
                    )
                    time.sleep(random.uniform(0.5, 1.0))
                    
                    # Hover over element (simulated by JavaScript)
                    driver.execute_script(
                        "arguments[0].dispatchEvent(new MouseEvent('mouseover', {bubbles: true}));", 
                        element
                    )
                    time.sleep(random.uniform(0.3, 0.7))
                    
                except Exception as e:
                    print(f"  - Error interacting with element: {e}")
                    continue
        
        # Perform smooth scrolling
        smooth_scroll_page(driver)
        
    except Exception as e:
        print(f"  - Error during page interaction: {e}")

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
    try:
        # Get the website index
        website_idx = WEBSITES.index(website_url)
        
        # Open target website in a new tab
        driver.switch_to.new_window('tab')
        driver.get(website_url)
        print(f"  - Opened target site: {website_url}")
        
        # Wait for initial page load and perform interactions
        time.sleep(random.uniform(3, 5))
        print("  - Starting page interaction...")
        interact_with_page(driver)
        print("  - Page interaction complete.")
        
        # Close the target website tab
        driver.close()
        
        # Switch back to the fingerprinting tab
        driver.switch_to.window(driver.window_handles[0])
        time.sleep(1)
        
        # Start trace collection
        print("  - Starting trace collection...")
        driver.execute_script(f"window.Alpine.store('app').collectTraceData('{website_url}', {website_idx})")
        
        # Wait for collection to finish
        try:
            wait.until(text_to_be_present_in_element_value(
                (By.ID, "status-message"), "Successfully processed trace"))
            print("  - Trace collected and saved successfully.")
            return True
        except TimeoutException:
            print("  - Timed out waiting for success message.")
            return False
            
    except Exception as e:
        print(f"  - Error during trace collection: {e}")
        # Ensure we're back on the main tab
        if len(driver.window_handles) > 1:
            driver.close()
        driver.switch_to.window(driver.window_handles[0])
        return False

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
        for website_url in WEBSITES:
            if current_counts.get(website_url, 0) < TRACES_PER_SITE:
                print(f"\nCollecting new trace for: {website_url}")
                success = collect_single_trace(driver, wait, website_url)
                
                if not success:
                    print("  - Collection failed, will retry later.")
                    time.sleep(2)  # Brief pause before next attempt
                    continue
                
                # Brief pause between collections
                time.sleep(2)
                
                # Clear previous results to avoid confusion
                try:
                    clear_trace_results(driver, wait)
                    print("  - Cleared previous results.")
                except Exception as e:
                    print(f"  - Warning: Could not clear results: {e}")
                
                break  # Move to next website after successful collection
        else:
            print("Collection seems complete, but loop continued. Exiting.")
            break

def main():
    if not is_server_running(port=5000):
        print("Error: Flask server is not running on port 5000.")
        print("Please start the server in a separate terminal with `python app.py` and try again.")
        sys.exit(1)

    database.db.init_database()

    # Self-healing loop: restart driver on crash
    while not is_collection_complete():
        driver = None
        try:
            print("\nSetting up new browser session...")
            driver = setup_webdriver()
            collect_fingerprints(driver)
        except Exception as e:
            print(f"\nAn error occurred, restarting collection: {type(e).__name__}")
            traceback.print_exc()  # Print full traceback for debugging
            time.sleep(5) # Pause before restarting
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
                print("Browser session closed.")
    
    print("\n--- Collection loop finished ---")

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
