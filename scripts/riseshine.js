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
    
    // Take initial screenshot
    if (process.env.GITHUB_ACTIONS) {
      await page.screenshot({ path: 'initial-page.png', fullPage: true });
      console.log('Initial page screenshot saved');
    }
    
    // Log the page title and URL
    const title = await page.title();
    const url = await page.url();
    console.log(`Page title: "${title}"`);
    console.log(`Page URL: ${url}`);
    
    // Get and log the page HTML structure
    const bodyHTML = await page.evaluate(() => {
      // Get first 2000 characters of body HTML for inspection
      return document.body ? document.body.innerHTML.substring(0, 2000) : 'No body element found';
    });
    console.log('=== PAGE HTML STRUCTURE (first 2000 chars) ===');
    console.log(bodyHTML);
    console.log('=== END HTML STRUCTURE ===');
    
    // Check for various possible error containers
    const errorSelectors = [
      '._errorContainer_2xb9v_8',
      '[class*="errorContainer"]',
      '[class*="error"]',
      'div:contains("sleep")',
      'div:contains("Zzzz")',
      'div:contains("inactivity")'
    ];
    
    let sleepContainer = null;
    let foundSelector = null;
    
    for (const selector of errorSelectors) {
      try {
        const element = await page.$(selector);
        if (element) {
          sleepContainer = element;
          foundSelector = selector;
          console.log(`Found sleep container with selector: ${selector}`);
          break;
        }
      } catch (err) {
        console.log(`Selector "${selector}" failed: ${err.message}`);
      }
    }
    
    if (sleepContainer) {
      console.log(`App is sleeping (found with: ${foundSelector}), looking for wake-up button...`);
      
      // Log all buttons on the page
      const allButtons = await page.evaluate(() => {
        const buttons = Array.from(document.querySelectorAll('button'));
        return buttons.map((btn, index) => ({
          index,
          innerHTML: btn.innerHTML.substring(0, 100),
          className: btn.className,
          id: btn.id,
          testId: btn.getAttribute('data-testid'),
          type: btn.type,
          visible: btn.offsetParent !== null
        }));
      });
      
      console.log('=== ALL BUTTONS ON PAGE ===');
      console.log(JSON.stringify(allButtons, null, 2));
      console.log('=== END BUTTONS ===');
      
      // Try multiple button selectors
      const buttonSelectors = [
        '[data-testid="wakeup-button-owner"]',
        'button[data-testid="wakeup-button-owner"]',
        '[data-testid*="wakeup"]',
        '[data-testid*="wake"]',
        'button:contains("Yes, get this app back up!")',
        'button:contains("wake")',
        'button:contains("back up")',
        '._restartButton_2xb9v_14',
        '[class*="restartButton"]',
        '[class*="button"][class*="primary"]'
      ];
      
      let wakeButton = null;
      let buttonSelector = null;
      
      for (const selector of buttonSelectors) {
        try {
          const element = await page.$(selector);
          if (element) {
            const isVisible = await element.isIntersectingViewport();
            console.log(`Found button with selector "${selector}", visible: ${isVisible}`);
            if (isVisible) {
              wakeButton = element;
              buttonSelector = selector;
              break;
            }
          }
        } catch (err) {
          console.log(`Button selector "${selector}" failed: ${err.message}`);
        }
      }
      
      if (wakeButton) {
        console.log(`Wake-up button found with selector: ${buttonSelector}, clicking...`);
        await wakeButton.click();
      } else {
        console.log('No wake-up button found, trying to wait for standard selector...');
        try {
          // Wait for the wake-up button specifically
          await page.waitForSelector('[data-testid="wakeup-button-owner"]', { 
            timeout: 10000,
            visible: true 
          });
          console.log('Standard wake-up button found after waiting, clicking...');
          await page.click('[data-testid="wakeup-button-owner"]');
        } catch (waitError) {
          console.log('Standard selector also failed, trying any button with wake-up text...');
          
          // Try clicking any button with wake-up text as last resort
          const textBasedClick = await page.evaluate(() => {
            const buttons = Array.from(document.querySelectorAll('button'));
            for (const btn of buttons) {
              if (btn.textContent && btn.textContent.toLowerCase().includes('get this app back up')) {
                btn.click();
                return true;
              }
            }
            return false;
          });
          
          if (!textBasedClick) {
            throw new Error('No wake-up button found with any method');
          }
          console.log('Clicked button using text-based search');
        }
      }
      
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
      await new Promise(resolve => setTimeout(resolve, 5000));
      
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