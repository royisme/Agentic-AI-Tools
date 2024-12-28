#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A more robust DuckDuckGo search script with optional JSON output, 
enhanced logging, and additional search parameters.
"""

import argparse
import sys
import logging
import json
import traceback
from duckduckgo_search import DDGS

def setup_logger(debug: bool = False):
    """
    Setup logging configuration.
    If debug=True, log level is DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def ddg_search(
    query: str,
    max_results: int = 10,
    region: str = "wt-wt",       # "wt-wt" = global (no region-specific results)
    safesearch: str = "moderate",# can be "off", "moderate", "strict"
    output_json: bool = False
):
    """
    Search using DuckDuckGo and return results with URLs and text snippets.
    Uses the HTML backend which has proven to be more reliable.
    
    Args:
        query (str): Search query
        max_results (int): Maximum number of results to return
        region (str): Region parameter for DuckDuckGo, e.g., "us-en", "wt-wt"
        safesearch (str): Safe search mode, e.g., "off", "moderate", "strict"
        output_json (bool): If True, prints results in JSON format; 
                            otherwise prints in a text/markdown style.

    Returns:
        None. (But prints results to stdout.)
    """
    logging.debug("Initiating DuckDuckGo search")
    logging.debug(f"Query: {query}")
    logging.debug(f"max_results: {max_results}, region: {region}, safesearch: {safesearch}, output_json: {output_json}")
    
    try:
        with DDGS() as ddgs:
            # backend='html' 仅使用 HTML 模式，据作者经验更稳定
            results = list(ddgs.text(
                query,
                region=region,
                safesearch=safesearch,
                max_results=max_results,
                backend='html'
            ))
            
            if not results:
                logging.warning("No results found.")
                return
            
            logging.info(f"Found {len(results)} results.")
            
            if output_json:
                # 如果需要 JSON 格式输出
                output = []
                for i, r in enumerate(results, 1):
                    output.append({
                        "rank": i,
                        "title": r.get('title', ''),
                        "url": r.get('link', r.get('href', 'N/A')),
                        "snippet": r.get('snippet', r.get('body', ''))
                    })
                print(json.dumps(output, indent=2, ensure_ascii=False))
            else:
                # 否则采用文本/Markdown 输出
                for i, r in enumerate(results, 1):
                    print(f"\n=== Result {i} ===")
                    print(f"URL: {r.get('link', r.get('href', 'N/A'))}")
                    print(f"Title: {r.get('title', 'N/A')}")
                    print(f"Snippet: {r.get('snippet', r.get('body', 'N/A'))}")

    except Exception as e:
        logging.error(f"Search failed due to error: {e}")
        logging.debug(f"Exception type: {type(e)}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="A DuckDuckGo search utility with optional JSON output and enhanced parameters.",
        epilog="Example usage:\n  python search.py 'openai gpt-4' --max-results 5 --json"
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument("--max-results", type=int, default=10,
                        help="Maximum number of results (default: 10)")
    parser.add_argument("--region", default="wt-wt",
                        help="Region for DuckDuckGo (e.g. us-en, wt-wt). Default: wt-wt (global).")
    parser.add_argument("--safesearch", default="moderate",
                        help="Safe search level: off, moderate, strict. Default: moderate.")
    parser.add_argument("--json", action="store_true",
                        help="If set, output results in JSON format.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug-level logging to stderr.")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logger(debug=args.debug)
    
    # 执行搜索
    ddg_search(
        query=args.query,
        max_results=args.max_results,
        region=args.region,
        safesearch=args.safesearch,
        output_json=args.json
    )

if __name__ == "__main__":
    main()

