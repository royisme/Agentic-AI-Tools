#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Playwright-based crawler for fetching web pages concurrently and extracting 
their text+links in Markdown or JSON format. Suitable for Agentic AI usage.
"""

import asyncio
import argparse
import sys
import os
from typing import List, Optional
import time
from urllib.parse import urlparse
import logging
import json

import html5lib
from playwright.async_api import async_playwright
from multiprocessing import Pool

# Configure base logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------
# 1. Async function to fetch page content
# ------------------------------------------------------------------------
async def fetch_page(url: str, context, timeout: int) -> Optional[str]:
    """
    Asynchronously fetch a webpage's content using a given browser context.

    Args:
        url (str): The URL to fetch.
        context: A Playwright browser context object.
        timeout (int): Maximum milliseconds to wait for loading and network idle.

    Returns:
        Optional[str]: The page's full HTML content if successful, else None.
    """
    page = await context.new_page()
    try:
        logger.debug(f"Fetching {url}")
        # 加速加载：阻止图片、CSS、字体等请求
        await page.route("**/*", lambda route: (
            route.abort() if route.request.resource_type in ["image", "stylesheet", "font"] 
            else route.continue_()
        ))

        # 设置超时
        await page.goto(url, timeout=timeout)
        await page.wait_for_load_state('networkidle', timeout=timeout)
        content = await page.content()
        logger.info(f"Successfully fetched {url}")
        return content
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return None
    finally:
        await page.close()


# ------------------------------------------------------------------------
# 2. Parsing function to extract text+links in Markdown
# ------------------------------------------------------------------------
def parse_html(html_content: Optional[str]) -> str:
    """
    Parse HTML content and extract text with hyperlinks in a Markdown-like format.
    Filters out scripts, styles, and some noise.

    Args:
        html_content (Optional[str]): The raw HTML to parse.

    Returns:
        str: Extracted text in a pseudo-Markdown style. Empty string if parse fails or html_content is None.
    """
    if not html_content:
        return ""

    try:
        document = html5lib.parse(html_content)
        result = []
        seen_texts = set()  # avoid printing duplicate texts

        def should_skip_element(elem) -> bool:
            """Check if the element should be skipped (scripts, styles, or whitespace-only)."""
            # Skip script and style tags
            if elem.tag in [
                '{http://www.w3.org/1999/xhtml}script',
                '{http://www.w3.org/1999/xhtml}style'
            ]:
                return True
            # Skip empty or whitespace-only elements
            if not any(text.strip() for text in elem.itertext()):
                return True
            return False

        def process_element(elem, depth=0):
            """Recursively process an element and its children, extracting text/links in markdown style."""
            if should_skip_element(elem):
                return

            # Handle text content in the current element
            if hasattr(elem, 'text') and elem.text:
                text = elem.text.strip()
                if text and text not in seen_texts:
                    # If it's an anchor tag, attempt to fetch the 'href'
                    if elem.tag == '{http://www.w3.org/1999/xhtml}a':
                        href = None
                        for attr, value in elem.items():
                            if attr.endswith('href'):
                                href = value
                                break
                        if href and not href.startswith(('#', 'javascript:')):
                            # Format as a markdown link
                            link_text = f"[{text}]({href})"
                            result.append("  " * depth + link_text)
                            seen_texts.add(text)
                    else:
                        result.append("  " * depth + text)
                        seen_texts.add(text)

            # Process children recursively
            for child in elem:
                process_element(child, depth + 1)

            # Handle tail text (the text that appears after an element tag)
            if hasattr(elem, 'tail') and elem.tail:
                tail = elem.tail.strip()
                if tail and tail not in seen_texts:
                    result.append("  " * depth + tail)
                    seen_texts.add(tail)

        # Start from body if possible
        body = document.find('.//{http://www.w3.org/1999/xhtml}body')
        if body is not None:
            process_element(body)
        else:
            # Fallback to processing the entire document
            process_element(document)

        # Filter out some known noise
        filtered_result = []
        noise_patterns = [
            'var ', 'function()', '.js', '.css', 'google-analytics',
            'disqus', '{', '}'  # too generic patterns can cause over-filtering
        ]
        for line in result:
            low_line = line.lower()
            if any(p in low_line for p in noise_patterns):
                continue
            filtered_result.append(line)

        return '\n'.join(filtered_result)

    except Exception as e:
        logger.error(f"Error parsing HTML: {str(e)}")
        return ""


# ------------------------------------------------------------------------
# 3. Master async function to process all URLs with concurrency
# ------------------------------------------------------------------------
async def process_urls(
    urls: List[str],
    max_concurrent: int = 5,
    timeout_ms: int = 30000,
    parallel_parse: bool = True,
    headless: bool = True
) -> List[str]:
    """
    Process multiple URLs concurrently using Playwright to fetch content, 
    then parse the HTML into text/markdown.

    Args:
        urls (List[str]): List of URLs to fetch.
        max_concurrent (int): Max number of concurrent browser contexts.
        timeout_ms (int): Timeout (in milliseconds) for each page load/network idle.
        parallel_parse (bool): Whether to parse HTML in parallel using multiprocessing pool.
        headless (bool): Whether to launch the browser in headless mode.

    Returns:
        List[str]: A list of parsed results (one per URL) in the same order as the input URLs.
    """
    logger.debug(f"Launching browser with headless={headless}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        try:
            # Create limited number of contexts
            n_contexts = min(len(urls), max_concurrent)
            contexts = [await browser.new_context() for _ in range(n_contexts)]

            # Create tasks for each URL
            tasks = []
            for i, url in enumerate(urls):
                context = contexts[i % len(contexts)]
                task = fetch_page(url, context, timeout=timeout_ms)
                tasks.append(task)

            # Gather raw HTML contents
            html_contents = await asyncio.gather(*tasks)

            # Either parse HTML in parallel or in the current process
            if parallel_parse:
                with Pool() as pool:
                    results = pool.map(parse_html, html_contents)
            else:
                results = list(map(parse_html, html_contents))

            return results

        finally:
            # Cleanup
            for context in contexts:
                await context.close()
            await browser.close()


# ------------------------------------------------------------------------
# 4. URL validator
# ------------------------------------------------------------------------
def validate_url(url: str) -> bool:
    """
    Validate if the given string is a well-formed URL.

    Args:
        url (str): The URL to validate.

    Returns:
        bool: True if valid, else False.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


# ------------------------------------------------------------------------
# 5. Command-line entry point
# ------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Fetch and extract text content from webpages using Playwright.',
        epilog='Example: python crawler.py https://example.com --headless --max-concurrent 5 --json'
    )
    parser.add_argument('urls', nargs='+', help='URLs to process')
    parser.add_argument('--max-concurrent', type=int, default=5,
                       help='Max number of concurrent browser contexts (default: 5)')
    parser.add_argument('--timeout-ms', type=int, default=30000,
                       help='Timeout in milliseconds for page load (default: 30000)')
    parser.add_argument('--headless', action='store_true',
                       help='Run the browser in headless mode (default off).')
    parser.add_argument('--json', action='store_true',
                       help='Output results in JSON format.')
    parser.add_argument('--no-parallel-parse', action='store_true',
                       help='Disable parallel parsing via multiprocessing pool.')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging.')

    args = parser.parse_args()

    # Adjust logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")

    # Validate URLs
    valid_urls = []
    for url in args.urls:
        if validate_url(url):
            valid_urls.append(url)
        else:
            logger.error(f"Invalid URL skipped: {url}")

    if not valid_urls:
        logger.error("No valid URLs provided. Exiting.")
        sys.exit(1)

    start_time = time.time()

    # Run main async process
    try:
        results = asyncio.run(process_urls(
            urls=valid_urls,
            max_concurrent=args.max_concurrent,
            timeout_ms=args.timeout_ms,
            parallel_parse=(not args.no_parallel_parse),
            headless=args.headless
        ))
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info(f"Total processing time: {elapsed:.2f}s")

    # Output
    if args.json:
        # Print results in JSON
        # Each entry: { "url": <url>, "content": <markdown> }
        output = []
        for url, text in zip(valid_urls, results):
            output.append({"url": url, "content": text})

        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        # Print results as raw text
        for url, text in zip(valid_urls, results):
            print(f"\n=== Content from {url} ===")
            print(text)
            print("=" * 80)


if __name__ == '__main__':
    main()

