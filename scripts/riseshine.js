const puppeteer = require('puppeteer');

async function wakeUpApp() {
  console.log('Starting wake-up process for xy-sonification.streamlit.app...');
  
  let browser;
  try {
    // Launch browser with GitHub Actions-friendly settings
    browser = await puppeteer.launch({
      headless: true, // GitHub Actions requires headless
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-accelerated-2d-canvas',
        '--no-first-run',
        '--no-zygote',
        '--disable-gpu'
      ]
    });

    const page = await browser.newPage();
    
    // Set viewport and user agent for better compatibility
    await page.setViewport({ width: 1280, height: 720 });
    await page.setUserAgent('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36');
    
    console.log('Navigating to Streamlit app...');
    
    // Navigate to the Streamlit app with extended timeout
    await page.goto('https://xy-sonification.streamlit.app/', {
      waitUntil: 'networkidle2',
      timeout: 60000 // 60 seconds timeout for slow cold starts
    });
    
    console.log('Page loaded, checking if app needs waking...');
    
    // Check if the sleeping page is displayed
    const sleepContainer = await page.$('._errorContainer_2xb9v_8');
    
    if (sleepContainer) {
      console.log('App is sleeping, looking for wake-up button...');
      
      // Wait for the wake-up button specifically
      await page.waitForSelector('[data-testid="wakeup-button-owner"]', { 
        timeout: 10000,
        visible: true 
      });
      
      console.log('Wake-up button found, clicking...');
      
      // Click the wake-up button
      await page.click('[data-testid="wakeup-button-owner"]');
      
      console.log('Wake-up button clicked, waiting for app to start...');
      
      // Wait for the app to wake up - look for the sleeping container to disappear
      // or for Streamlit app elements to appear
      try {
        await page.waitForFunction(
          () => !document.querySelector('._errorContainer_2xb9v_8') || 
                document.querySelector('[data-testid="stApp"]'),
          { timeout: 120000 } // 2 minutes for cold start
        );
        console.log('App appears to be waking up...');
      } catch (timeoutError) {
        console.log('Timeout waiting for app to wake, but button was clicked');
      }
      
      // Additional wait to ensure the app is fully loaded
      await page.waitForTimeout(5000);
      
    } else {
      console.log('App appears to already be awake (no sleep container found)');
      
      // Try to wait for normal Streamlit elements to confirm it's working
      try {
        await page.waitForSelector('[data-testid="stApp"]', { timeout: 30000 });
        console.log('Streamlit app is running normally');
      } catch (err) {
        console.log('Could not detect normal Streamlit elements, but no sleep screen either');
      }
    }
    
    // Take a screenshot for debugging if in GitHub Actions
    if (process.env.GITHUB_ACTIONS) {
      await page.screenshot({ 
        path: 'wake-up-result.png', 
        fullPage: false 
      });
      console.log('Screenshot saved for debugging');
    }
    
    console.log('Wake-up process completed successfully!');
    
  } catch (error) {
    console.error('Error during wake-up process:', error);
    
    // Take error screenshot if possible
    if (browser && process.env.GITHUB_ACTIONS) {
      try {
        const page = await browser.newPage();
        await page.screenshot({ path: 'wake-up-error.png' });
        console.log('Error screenshot saved');
      } catch (screenshotError) {
        console.log('Could not save error screenshot');
      }
    }
    
    throw error;
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

// Run the wake-up function
wakeUpApp()
  .then(() => {
    console.log('✅ App wake-up completed');
    process.exit(0);
  })
  .catch((error) => {
    console.error('❌ App wake-up failed:', error);
    process.exit(1);
  });