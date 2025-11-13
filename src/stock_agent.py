"""
Stock Correlation Analysis Agent

LLM-powered agent for selecting correlated stocks across multiple markets.
Supports Google AI (Gemini), OpenAI, DeepSeek, and Anthropic Claude.
"""

import json
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import re


class StockCorrelationAgent:
    """
    Intelligent agent for analyzing industry sectors and selecting
    correlated stocks across multiple markets.

    Supports:
    - Google AI (Gemini) - Default for Colab Pro+
    - OpenAI (GPT-4)
    - DeepSeek
    - Anthropic (Claude)
    """

    def __init__(
        self,
        industry: str,
        markets: List[str],
        min_stocks_per_market: Dict[str, int],
        llm_provider: str = 'google',
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize the stock analysis agent.

        Args:
            industry: Industry sector (e.g., "Semiconductor", "Automotive")
            markets: List of markets (e.g., ["US", "JP", "CN", "HK"])
            min_stocks_per_market: Dict of minimum stocks per market
            llm_provider: 'google', 'openai', 'deepseek', or 'anthropic'
            api_key: API key for the LLM provider
            model_name: Optional custom model name
        """
        self.industry = industry
        self.markets = markets
        self.min_stocks_per_market = min_stocks_per_market
        self.llm_provider = llm_provider.lower()
        self.api_key = api_key

        # Default model names
        self.model_names = {
            'google': 'gemini-1.5-pro',
            'openai': 'gpt-4-turbo-preview',
            'deepseek': 'deepseek-chat',
            'anthropic': 'claude-3-5-sonnet-20241022'
        }

        self.model_name = model_name or self.model_names.get(self.llm_provider)

        # Initialize LLM client
        self.client = self._init_llm_client()

        # Results
        self.selected_stocks = None
        self.analysis_report = None

    def _init_llm_client(self):
        """Initialize the appropriate LLM client."""
        if self.llm_provider == 'google':
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                return genai.GenerativeModel(self.model_name)
            except ImportError:
                raise ImportError("Please install: pip install google-generativeai")

        elif self.llm_provider == 'openai':
            try:
                from openai import OpenAI
                return OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install: pip install openai")

        elif self.llm_provider == 'deepseek':
            try:
                from openai import OpenAI
                return OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.deepseek.com"
                )
            except ImportError:
                raise ImportError("Please install: pip install openai")

        elif self.llm_provider == 'anthropic':
            try:
                from anthropic import Anthropic
                return Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install: pip install anthropic")

        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _generate_prompt(self) -> str:
        """Generate the analysis prompt for the LLM."""
        prompt = f"""You are a stock correlation analysis expert specializing in cross-market relationships.

Industry Sector: {self.industry}
Target Markets: {', '.join(self.markets)}
Minimum stocks per market: {self.min_stocks_per_market}

Task: Identify stocks in the specified markets that are likely to have correlated price movements within this industry sector.

Consider these factors:
1. Direct competitors in the same industry
2. Supply chain relationships (upstream suppliers, downstream customers)
3. Major industry indices and sector ETFs
4. Companies affected by the same macro factors (e.g., commodity prices, regulations)
5. Cross-market listings (e.g., ADRs for Chinese companies)

Requirements:
- Each market must have AT LEAST the minimum number of stocks specified
- Prioritize liquid stocks with high trading volume
- Include major indices/ETFs for each market
- For each stock, provide: ticker symbol, company name, and selection reason
- Use correct ticker formats:
  * US: Standard ticker (e.g., "AAPL", "NVDA")
  * Japan: Ticker with .T suffix (e.g., "6758.T" for Sony)
  * China A-share: 6-digit code (e.g., "688981" for SMIC)
  * Hong Kong: 5-digit code with .HK suffix (e.g., "00700.HK" for Tencent)

Output ONLY a valid JSON object in this exact format (no markdown, no extra text):
{{
  "metadata": {{
    "industry": "{self.industry}",
    "analysis_date": "{datetime.now().strftime('%Y-%m-%d')}",
    "total_stocks": <number>
  }},
  "stocks": {{
    "US": [
      {{"ticker": "TICKER", "name": "Company Name", "reason": "Selection reason", "relevance_score": 0.95}},
      ...
    ],
    "JP": [...],
    "CN": [...],
    "HK": [...]
  }},
  "relationships": {{
    "supply_chain": [
      {{"supplier": "TICKER1", "customer": "TICKER2", "relationship": "Description"}}
    ],
    "competitors": [
      {{"stocks": ["TICKER1", "TICKER2"], "segment": "Market segment"}}
    ]
  }}
}}

Example for Semiconductor industry:
{{
  "metadata": {{
    "industry": "Semiconductor",
    "analysis_date": "2024-01-15",
    "total_stocks": 23
  }},
  "stocks": {{
    "US": [
      {{"ticker": "NVDA", "name": "NVIDIA Corporation", "reason": "Leading GPU manufacturer for AI and gaming", "relevance_score": 0.98}},
      {{"ticker": "AMD", "name": "Advanced Micro Devices", "reason": "Direct competitor in GPU/CPU markets", "relevance_score": 0.95}},
      {{"ticker": "INTC", "name": "Intel Corporation", "reason": "Major CPU manufacturer and foundry player", "relevance_score": 0.92}},
      {{"ticker": "SMH", "name": "VanEck Semiconductor ETF", "reason": "Broad semiconductor sector index", "relevance_score": 0.90}}
    ],
    "JP": [
      {{"ticker": "8035.T", "name": "Tokyo Electron Limited", "reason": "Chip equipment manufacturer, benefits from global capacity expansion", "relevance_score": 0.88}},
      {{"ticker": "6758.T", "name": "Sony Group Corporation", "reason": "Image sensors and gaming chips", "relevance_score": 0.82}}
    ]
  }},
  "relationships": {{
    "supply_chain": [
      {{"supplier": "ASML", "customer": "TSMC", "relationship": "Equipment supplier for chip manufacturing"}}
    ],
    "competitors": [
      {{"stocks": ["NVDA", "AMD"], "segment": "GPU market"}}
    ]
  }}
}}

Now analyze the {self.industry} industry for the markets: {', '.join(self.markets)}"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        if self.llm_provider == 'google':
            response = self.client.generate_content(prompt)
            return response.text

        elif self.llm_provider in ['openai', 'deepseek']:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial analysis expert. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content

        elif self.llm_provider == 'anthropic':
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.content[0].text

        else:
            raise ValueError(f"Unsupported provider: {self.llm_provider}")

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Remove markdown code blocks if present
        response = response.strip()
        if response.startswith('```'):
            # Extract JSON from markdown code block
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
            else:
                # Try to find JSON without code blocks
                response = re.sub(r'^```(?:json)?', '', response)
                response = re.sub(r'```$', '', response)

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response: {response[:500]}...")
            raise

    def _validate_stocks(self, stocks_data: Dict) -> Dict:
        """Validate and ensure minimum stock requirements."""
        validated = stocks_data.copy()

        for market in self.markets:
            if market not in validated['stocks']:
                validated['stocks'][market] = []

            current_count = len(validated['stocks'][market])
            min_required = self.min_stocks_per_market.get(market, 3)

            if current_count < min_required:
                print(f"⚠️ Warning: {market} has only {current_count} stocks (min: {min_required})")

        return validated

    def _generate_markdown_report(self, stocks_data: Dict) -> str:
        """Generate a detailed markdown analysis report."""
        report = f"""# Stock Correlation Analysis Report

**Industry**: {self.industry}
**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**LLM Model**: {self.llm_provider.upper()} - {self.model_name}
**Markets**: {', '.join(self.markets)}

---

## 📊 Summary

"""

        # Summary statistics
        total_stocks = sum(len(stocks_data['stocks'].get(m, [])) for m in self.markets)
        report += f"- **Total Stocks Selected**: {total_stocks}\n"

        for market in self.markets:
            count = len(stocks_data['stocks'].get(market, []))
            report += f"- **{market} Market**: {count} stocks\n"

        report += "\n---\n\n"

        # Detailed analysis by market
        market_flags = {
            'US': '🇺🇸',
            'JP': '🇯🇵',
            'CN': '🇨🇳',
            'HK': '🇭🇰'
        }

        for market in self.markets:
            flag = market_flags.get(market, '🌍')
            stocks = stocks_data['stocks'].get(market, [])

            report += f"## {flag} {market} Market ({len(stocks)} stocks)\n\n"

            for i, stock in enumerate(stocks, 1):
                ticker = stock.get('ticker', 'N/A')
                name = stock.get('name', 'N/A')
                reason = stock.get('reason', 'N/A')
                score = stock.get('relevance_score', 0)
                stars = '⭐' * min(5, int(score * 5))

                report += f"### {i}. **{ticker}** - {name}\n"
                report += f"- **Reason**: {reason}\n"
                report += f"- **Relevance**: {stars} ({score:.2f})\n\n"

            report += "---\n\n"

        # Relationships
        if 'relationships' in stocks_data:
            report += "## 🔗 Cross-Market Relationships\n\n"

            if 'supply_chain' in stocks_data['relationships']:
                report += "### Supply Chain\n\n"
                for rel in stocks_data['relationships']['supply_chain']:
                    supplier = rel.get('supplier', 'N/A')
                    customer = rel.get('customer', 'N/A')
                    relationship = rel.get('relationship', 'N/A')
                    report += f"- **{supplier}** → **{customer}**: {relationship}\n"
                report += "\n"

            if 'competitors' in stocks_data['relationships']:
                report += "### Competitive Dynamics\n\n"
                for comp in stocks_data['relationships']['competitors']:
                    stocks_list = ', '.join(comp.get('stocks', []))
                    segment = comp.get('segment', 'N/A')
                    report += f"- **{segment}**: {stocks_list}\n"
                report += "\n"

        report += "---\n\n"
        report += "## 💡 Prediction Strategy Recommendations\n\n"

        # Add strategy recommendations based on markets
        if 'US' in self.markets and 'JP' in self.markets:
            report += "### US → JP (Expected High Correlation ⭐⭐⭐⭐⭐)\n"
            report += "- **Logic**: US market movements (especially tech) often lead Japanese markets\n"
            report += "- **Time advantage**: US closes, 3-hour window, JP opens\n"
            report += "- **Expected Sharpe**: 1.5-2.0\n\n"

        if 'US' in self.markets and 'CN' in self.markets:
            report += "### US → CN (Expected Medium Correlation ⭐⭐⭐)\n"
            report += "- **Logic**: Global supply chain effects, but policy interference\n"
            report += "- **Challenges**: Capital controls, different market dynamics\n"
            report += "- **Expected Sharpe**: 0.8-1.2\n\n"

        if 'US' in self.markets and 'HK' in self.markets:
            report += "### US → HK (Expected Good Correlation ⭐⭐⭐⭐)\n"
            report += "- **Logic**: Open market with many ADR arbitrage opportunities\n"
            report += "- **Advantage**: Chinese tech stocks with US listings\n"
            report += "- **Expected Sharpe**: 1.2-1.8\n\n"

        report += "---\n\n"
        report += "*This report was generated by an AI agent and should be used for reference only. Always conduct your own research before making investment decisions.*\n"

        return report

    def analyze(self) -> Tuple[Dict, str]:
        """
        Run the complete analysis.

        Returns:
            (stocks_json, markdown_report)
        """
        print(f"🤖 Analyzing {self.industry} industry...")
        print(f"📍 Markets: {', '.join(self.markets)}")
        print(f"🧠 Using {self.llm_provider.upper()} - {self.model_name}")
        print()

        # Generate prompt
        prompt = self._generate_prompt()

        # Call LLM
        print("💭 Querying LLM...")
        response = self._call_llm(prompt)

        # Parse response
        print("📊 Parsing response...")
        stocks_data = self._parse_json_response(response)

        # Validate
        print("✅ Validating results...")
        stocks_data = self._validate_stocks(stocks_data)

        # Generate report
        print("📝 Generating report...")
        report = self._generate_markdown_report(stocks_data)

        self.selected_stocks = stocks_data
        self.analysis_report = report

        # Print summary
        total = sum(len(stocks_data['stocks'].get(m, [])) for m in self.markets)
        print(f"\n✅ Analysis complete!")
        print(f"📊 Selected {total} stocks across {len(self.markets)} markets")

        return stocks_data, report

    def save_results(self, output_dir: str = 'outputs'):
        """Save analysis results to files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save JSON
        json_path = os.path.join(output_dir, 'stocks_selection.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.selected_stocks, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved: {json_path}")

        # Save report
        report_path = os.path.join(output_dir, 'analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.analysis_report)

        print(f"💾 Saved: {report_path}")

        return json_path, report_path


# ============================================================================
# Utility Functions
# ============================================================================

def load_stocks_from_json(json_path: str) -> Dict:
    """Load stock selection from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_tickers_by_market(stocks_data: Dict, market: str) -> List[str]:
    """Extract ticker symbols for a specific market."""
    stocks = stocks_data.get('stocks', {}).get(market, [])
    return [stock['ticker'] for stock in stocks]


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    """Example: Analyze semiconductor industry"""

    # Configuration
    config = {
        'industry': 'Semiconductor',
        'markets': ['US', 'JP'],
        'min_stocks_per_market': {
            'US': 6,
            'JP': 4
        },
        'llm_provider': 'google',  # or 'openai', 'deepseek', 'anthropic'
        'api_key': os.getenv('GOOGLE_AI_API_KEY')  # or from user input
    }

    # Create agent
    agent = StockCorrelationAgent(**config)

    # Run analysis
    stocks_json, report = agent.analyze()

    # Save results
    agent.save_results()

    # Print report preview
    print("\n" + "="*60)
    print("ANALYSIS REPORT (Preview)")
    print("="*60)
    print(report[:1000] + "...\n")

    # Show selected tickers
    print("Selected US tickers:", get_tickers_by_market(stocks_json, 'US'))
    print("Selected JP tickers:", get_tickers_by_market(stocks_json, 'JP'))
