const puppeteer = require('puppeteer');

(async () => {
  console.log('Testing Puppeteer with Chromium...');
  
  try {
    const browser = await puppeteer.launch({
      headless: true,
      executablePath: '/usr/bin/chromium-browser',
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-gpu'
      ]
    });
    
    console.log('Browser launched successfully!');
    
    const page = await browser.newPage();
    await page.goto('https://example.com');
    
    const title = await page.title();
    console.log('Page title:', title);
    
    const content = await page.content();
    console.log('Page loaded, content length:', content.length);
    
    await browser.close();
    console.log('Test completed successfully!');
  } catch (error) {
    console.error('Error:', error.message);
  }
})();