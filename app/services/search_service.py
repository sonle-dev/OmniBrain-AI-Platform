import aiohttp
import logging
from typing import Dict, Any, List, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class SearchService:
    @staticmethod
    def _lazy_import_tavily():
        """Lazy loads the Tavily client to avoid startup overhead."""
        from tavily import TavilyClient
        return TavilyClient

    async def search_web(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Searches the web for the given query using Tavily API.
        Falls back to a public search API (DuckDuckGo JSON endpoint) if Tavily API Key is not set.
        """
        # 1. Check Tavily configuration
        if settings.TAVILY_API_KEY:
            try:
                TavilyClient = self._lazy_import_tavily()
                client = TavilyClient(api_key=settings.TAVILY_API_KEY)
                
                # Run the blocking SDK call in an async-friendly executor
                import asyncio
                response = await asyncio.to_thread(
                    client.search,
                    query=query,
                    max_results=max_results,
                    search_depth="advanced"
                )
                
                results = []
                for item in response.get("results", []):
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "content": item.get("content", "")
                    })
                
                return {
                    "source": "tavily",
                    "query": query,
                    "results": results
                }
            except Exception as e:
                logger.error(f"Tavily search failed: {e}. Attempting fallback...")

        # 2. Fallback: DuckDuckGo Instant Answer / HTML Search
        try:
            logger.info("Attempting fallback search via DuckDuckGo...")
            url = "https://html.duckduckgo.com/html/"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            }
            data = {"q": query}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data, headers=headers, timeout=5) as resp:
                    if resp.status == 200:
                        html = await resp.text()
                        
                        # Lazy import BeautifulSoup to parse HTML
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(html, "html.parser")
                        
                        results = []
                        links = soup.find_all("a", class_="result__url")
                        snippets = soup.find_all("a", class_="result__snippet")
                        titles = soup.find_all("a", class_="result__snippet") # fallback for title
                        
                        # Find main search result divs
                        elements = soup.find_all("div", class_="result")
                        for element in elements[:max_results]:
                            title_a = element.find("a", class_="result__a")
                            snippet_a = element.find("a", class_="result__snippet")
                            if title_a:
                                title = title_a.get_text(strip=True)
                                href = title_a.get("href", "")
                                # Fix DuckDuckGo redirect URLs
                                if href.startswith("//duckduckgo.com/y.js"):
                                    # Extract real URL from query parameters or skip
                                    pass
                                snippet = snippet_a.get_text(strip=True) if snippet_a else ""
                                results.append({
                                    "title": title,
                                    "url": href,
                                    "content": snippet
                                })
                                
                        if results:
                            return {
                                "source": "duckduckgo_fallback",
                                "query": query,
                                "results": results
                            }
        except Exception as fallback_err:
            logger.error(f"Fallback search failed: {fallback_err}")
            
        # 3. Double Fallback: Return a mock informative result
        return {
            "source": "mock_fallback",
            "query": query,
            "results": [
                {
                    "title": "Configuring Tavily API Key",
                    "url": "https://tavily.com",
                    "content": (
                        "OmniBrain Web Search is currently in fallback mode. Please configure "
                        "TAVILY_API_KEY in your .env file to enable high-quality live web search."
                    )
                }
            ]
        }

# Singleton Instance
search_service = SearchService()
