"""
Web Scraping Function for RAG Collections
Simple, effective web content extraction
"""

import asyncio
import hashlib
import logging
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse
import re

import httpx
from bs4 import BeautifulSoup
from trafilatura import extract
from trafilatura.settings import use_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text content
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Import html module for entity decoding
    import html
    
    # Decode HTML entities (&nbsp;, &amp;, etc.)
    text = html.unescape(text)
    
    # Remove any residual HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Remove JavaScript and CSS
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove URLs (optional - comment out if you want to keep URLs)
    # text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses (optional)
    # text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Remove special characters and symbols (keep basic punctuation)
    text = re.sub(r'[^\w\s.,!?;:\-\'"()\[\]{}/@#$%&*+=]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # Multiple newlines to double newline
    
    # Remove leading/trailing whitespace from each line
    text = '\n'.join(line.strip() for line in text.split('\n'))
    
    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove common artifacts
    text = re.sub(r'[\u200b-\u200f\u202a-\u202e]', '', text)
    text = re.sub(r'\u00a0', ' ', text)
    
    
    patterns_to_remove = [
        r'Skip to (main )?content',
        r'Cookie (Policy|Settings|Preferences)',
        r'Accept (all )?cookies',
        r'Privacy Policy',
        r'Terms (of Service|and Conditions)',
        r'Subscribe to (our )?newsletter',
        r'Follow us on',
        r'Share on (Facebook|Twitter|LinkedIn)',
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()


async def get_links(url: str, base_domain: str, ignore_ids: bool = True):
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return set()

            html = resp.text

        soup = BeautifulSoup(html, "html.parser")
        links = set()

        for tag in soup.find_all("a", href=True):
            href = tag["href"]

            # Remove #fragment if needed
            if ignore_ids:
                href = href.split("#")[0]

            if not href:
                continue

            # Convert relative → absolute
            absolute_url = urljoin(url, href)

            # Check same domain
            if urlparse(absolute_url).hostname == base_domain:
                links.add(absolute_url)

        return links

    except Exception:
        print(f"Failed to fetch: {url}")
        return set()

async def crawl(
    start_url: str,
    max_depth: int = 2,
    max_links: int = 5,
    ignore_ids: bool = True
):
    """
    Async DFS crawler that collects up to `max_links` URLs 
    within `max_depth` depth from the start_url.
    """

    if get_links is None:
        raise ValueError("crawl() requires an async get_links(url, base_domain, ignore_ids) function")

    base_domain = urlparse(start_url).hostname
    to_visit = [{"url": start_url, "depth": 0}]
    visited = set()

    while to_visit and len(visited) < max_links:
        node = to_visit.pop()
        url, depth = node["url"], node["depth"]

        if url in visited:
            continue

        visited.add(url)

        # stop if depth reached
        if depth >= max_depth:
            continue

        # fetch outgoing links (async)
        links = await get_links(url, base_domain, ignore_ids)

        # push new links to stack
        for link in links:
            if link not in visited and len(visited) < max_links:
                to_visit.append({"url": link, "depth": depth + 1})

    return list(visited)


async def scrape_website(
    url: str,
    timeout: int = 30,
    max_retries: int = 3,
    user_agent: Optional[str] = None
) -> Optional[Dict[str, str]]:
    """
    Scrape a website and extract clean text content
    
    Args:
        url: URL to scrape
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        user_agent: Custom user agent (optional)
    
    Returns:
        Dictionary containing:
            - url: Original URL
            - doc_id: Unique document ID
            - content: Extracted text content
            - title: Page title
            - description: Meta description
            - domain: Domain name
            - content_length: Length of content
        
        Returns None if scraping fails
    
    Example:
        result = await scrape_website("https://example.com")
        if result:
            print(f"Title: {result['title']}")
            print(f"Content: {result['content'][:200]}...")
    """
    
    # Validate URL
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        logger.error(f"Invalid URL: {url}")
        return None
    
    # Default user agent
    if not user_agent:
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    
    # Fetch the page with retries
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers=headers
    ) as client:
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching {url} (attempt {attempt + 1}/{max_retries})")
                response = await client.get(url)
                response.raise_for_status()
                html = response.text
                break
                
            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP {e.response.status_code} for {url}")
                if e.response.status_code in [404, 403, 401]:
                    return None
                if attempt == max_retries - 1:
                    return None
                    
            except httpx.RequestError as e:
                logger.warning(f"Request error: {str(e)}")
                if attempt == max_retries - 1:
                    return None
                    
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                if attempt == max_retries - 1:
                    return None
            
            # Exponential backoff
            await asyncio.sleep(2 ** attempt)
        else:
            return None
    
    # Extract content using trafilatura (best for articles)
    try:
        config = use_config()
        config.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")
        
        content = extract(
            html,
            config=config,
            include_comments=False,
            include_tables=True,
            include_images=False,
            output_format='txt',
            url=url
        )
    except Exception as e:
        logger.warning(f"Trafilatura failed: {str(e)}")
        content = None
    
    # Fallback to BeautifulSoup if trafilatura fails
    if not content:
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()
            
            # Try to find main content
            main_content = (
                soup.find('main') or 
                soup.find('article') or 
                soup.find('div', {'role': 'main'}) or
                soup.find('div', {'class': re.compile(r'content|main|article', re.I)}) or
                soup.body or 
                soup
            )
            
            # Extract text
            content = main_content.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            content = re.sub(r'\n\s*\n+', '\n\n', content)
            content = re.sub(r' +', ' ', content)
            
        except Exception as e:
            logger.error(f"BeautifulSoup extraction failed: {str(e)}")
            return None
    
    if not content or len(content.strip()) < 100:
        logger.error(f"Insufficient content extracted from {url}")
        return None
    
    # Extract metadata
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Get title
        title = None
        if soup.title:
            title = soup.title.string.strip()
        elif soup.find('h1'):
            title = soup.find('h1').get_text(strip=True)
        
        # Get description
        description = None
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            description = meta_desc['content'].strip()
        
        # Get author if available
        author = None
        meta_author = soup.find('meta', attrs={'name': 'author'})
        if meta_author and meta_author.get('content'):
            author = meta_author['content'].strip()
            
    except Exception as e:
        logger.warning(f"Metadata extraction failed: {str(e)}")
        title = description = author = None
    
    # Generate unique document ID
    doc_id = hashlib.sha256(url.encode()).hexdigest()[:16]
    
    # Build result
    result = {
        "url": url,
        "doc_id": doc_id,
        "content": clean_text(content.strip()),
        "title": title or "",
        "description": description or "",
        "author": author or "",
        "domain": parsed.netloc,
        "content_length": len(content.strip())
    }
    
    logger.info(f"Successfully scraped {url} ({result['content_length']} chars)")
    return result


async def scrape_multiple_websites(
    urls: List[str],
    max_concurrent: int = 5,
    **kwargs
) -> List[Dict[str, str]]:
    """
    Scrape multiple websites concurrently
    
    Args:
        urls: List of URLs to scrape
        max_concurrent: Maximum concurrent requests (default: 5)
        **kwargs: Additional arguments passed to scrape_website()
    
    Returns:
        List of successfully scraped documents
    
    Example:
        urls = [
            "https://example.com",
            "https://another-site.com",
            "https://blog.example.org"
        ]
        results = await scrape_multiple_websites(urls)
        print(f"Scraped {len(results)}/{len(urls)} websites")
    """
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def scrape_with_limit(url: str):
        async with semaphore:
            return await scrape_website(url, **kwargs)
    
    tasks = [scrape_with_limit(url) for url in urls]
    results = await asyncio.gather(*tasks)
    
    # Filter out failed scrapes
    successful = [doc for doc in results if doc is not None]
    
    logger.info(f"Successfully scraped {len(successful)}/{len(urls)} websites")
    return successful

if __name__ == "__main__":

    async def test():
        test_url = "https://aisensy.com/"
        list_of_links = await crawl(test_url, max_depth=2, max_links=3, ignore_ids=True)
        results = await scrape_multiple_websites(list_of_links)
        for result in results:
            if result:
                print(f"Title: {result['title']}")
                print(f"Content: {result['content'][:200]}...")
        else:
            print("Scraping failed.")

    asyncio.run(test())