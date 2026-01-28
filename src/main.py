"""CLI entry point for TASE Stock AI Agent."""

import asyncio
import logging
import sys
from datetime import datetime
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich import box

from src.config import get_settings
from src.agent import StockAgent
from src.rules.scoring import Recommendation
from src.rules.strategies import RiskLevel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rich console
console = Console()


def get_recommendation_color(rec: Recommendation) -> str:
    """Get color for recommendation."""
    colors = {
        Recommendation.STRONG_BUY: "bold green",
        Recommendation.BUY: "green",
        Recommendation.HOLD: "yellow",
        Recommendation.SELL: "red",
        Recommendation.STRONG_SELL: "bold red",
    }
    return colors.get(rec, "white")


def get_score_color(score: float) -> str:
    """Get color for score."""
    if score >= 80:
        return "bold green"
    elif score >= 60:
        return "green"
    elif score >= 40:
        return "yellow"
    elif score >= 20:
        return "red"
    else:
        return "bold red"


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug logging')
def cli(debug: bool):
    """TASE Stock AI Agent - Israeli Stock Exchange Recommendations.
    
    An AI-powered stock analysis tool for the Tel Aviv Stock Exchange,
    optimized for swing trades (1-4 week holding period).
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument('symbol')
@click.option('--no-ai', is_flag=True, help='Skip AI analysis')
def analyze(symbol: str, no_ai: bool):
    """Analyze a specific stock.
    
    Example: python -m src.main analyze TEVA
    """
    settings = get_settings()
    
    # Check configuration
    if not no_ai and not settings.gemini_api_key:
        console.print("[yellow]Warning: GEMINI_API_KEY not set. AI analysis disabled.[/yellow]")
        no_ai = True
    
    async def run_analysis():
        agent = StockAgent(settings)
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description=f"Analyzing {symbol}...", total=None)
                analysis, ai_analysis = await agent.analyze_stock(symbol, include_ai=not no_ai)
            
            # Display results
            console.print()
            
            # Header panel
            rec_color = get_recommendation_color(analysis.recommendation)
            header = Panel(
                f"[bold]{analysis.name}[/bold]\n"
                f"Symbol: {analysis.symbol} | Price: ₪{analysis.current_price:.2f}\n"
                f"Recommendation: [{rec_color}]{analysis.recommendation.value}[/{rec_color}] | "
                f"Confidence: {analysis.confidence:.0f}%",
                title=f"Stock Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                border_style="blue",
            )
            console.print(header)
            
            # Scores table
            score_table = Table(title="Analysis Scores", box=box.ROUNDED)
            score_table.add_column("Category", style="cyan")
            score_table.add_column("Score", justify="right")
            score_table.add_column("Weight", justify="right")
            
            score_table.add_row(
                "Technical",
                f"[{get_score_color(analysis.technical_score)}]{analysis.technical_score:.1f}[/]",
                "40%"
            )
            score_table.add_row(
                "Fundamental",
                f"[{get_score_color(analysis.fundamental_score)}]{analysis.fundamental_score:.1f}[/]",
                "35%"
            )
            score_table.add_row(
                "Sentiment",
                f"[{get_score_color(analysis.sentiment_score)}]{analysis.sentiment_score:.1f}[/]",
                "25%"
            )
            score_table.add_row(
                "[bold]Total[/bold]",
                f"[{get_score_color(analysis.total_score)}][bold]{analysis.total_score:.1f}[/bold][/]",
                "100%"
            )
            
            console.print(score_table)
            console.print()
            
            # Signals
            if analysis.bullish_signals:
                console.print("[green]Bullish Signals:[/green]")
                for signal in analysis.bullish_signals:
                    console.print(f"  [green]✓[/green] {signal}")
                console.print()
            
            if analysis.bearish_signals:
                console.print("[red]Bearish Signals:[/red]")
                for signal in analysis.bearish_signals:
                    console.print(f"  [red]✗[/red] {signal}")
                console.print()
            
            # Targets
            if analysis.target_price or analysis.stop_loss:
                console.print("[bold]Trading Levels:[/bold]")
                if analysis.target_price:
                    console.print(f"  Target: ₪{analysis.target_price:.2f}")
                if analysis.stop_loss:
                    console.print(f"  Stop Loss: ₪{analysis.stop_loss:.2f}")
                console.print(f"  Holding Period: {analysis.holding_period}")
                console.print()
            
            # AI Analysis
            if ai_analysis:
                console.print(Panel(
                    ai_analysis.summary,
                    title="AI Analysis",
                    border_style="magenta",
                ))
                
                if ai_analysis.key_factors:
                    console.print("[bold]Key Factors:[/bold]")
                    for factor in ai_analysis.key_factors:
                        console.print(f"  • {factor}")
                
                if ai_analysis.risks:
                    console.print("\n[bold]Risks:[/bold]")
                    for risk in ai_analysis.risks:
                        console.print(f"  • {risk}")
                
                if ai_analysis.catalysts:
                    console.print("\n[bold]Catalysts:[/bold]")
                    for catalyst in ai_analysis.catalysts:
                        console.print(f"  • {catalyst}")
            
        finally:
            await agent.close()
    
    asyncio.run(run_analysis())


@cli.command()
@click.option('--min-score', default=60, help='Minimum score threshold')
@click.option('--top', default=10, help='Number of top picks to show')
@click.option('--no-ai', is_flag=True, help='Skip AI analysis')
@click.option('--all', 'scan_all', is_flag=True, help='Scan ALL TASE stocks dynamically (slower but comprehensive)')
@click.option('--max-stocks', default=100, help='Maximum stocks to scan when using --all')
def scan(min_score: float, top: int, no_ai: bool, scan_all: bool, max_stocks: int):
    """Scan TASE stocks and get recommendations.
    
    By default, scans an extended watchlist of ~100 stocks.
    Use --all to dynamically fetch and scan ALL stocks from the TASE exchange.
    
    Examples:
    
        python -m src.main scan                    # Scan extended watchlist
        
        python -m src.main scan --all              # Scan ALL TASE stocks
        
        python -m src.main scan --all --max-stocks 200   # Scan up to 200 stocks
        
        python -m src.main scan --min-score 70 --top 5   # Higher threshold
    """
    settings = get_settings()
    
    if not no_ai and not settings.gemini_api_key:
        console.print("[yellow]Warning: GEMINI_API_KEY not set. AI analysis disabled.[/yellow]")
        no_ai = True
    
    async def run_scan():
        agent = StockAgent(settings)
        
        try:
            # Show what we're doing
            if scan_all:
                console.print(f"[bold cyan]Scanning ALL TASE stocks dynamically (max {max_stocks})...[/bold cyan]")
            else:
                console.print("[bold cyan]Scanning extended TASE watchlist...[/bold cyan]")
            console.print()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(description="Fetching stocks and analyzing...", total=None)
                result = await agent.scan_market(
                    min_score=min_score,
                    include_ai=not no_ai,
                    top_n=top,
                    scan_all=scan_all,
                    max_stocks=max_stocks,
                )
            
            console.print()
            
            # Scan statistics
            stats_panel = Panel(
                f"[bold]Source:[/bold] {result.source}\n"
                f"[bold]Stocks Scanned:[/bold] {result.total_scanned}\n"
                f"[bold]Meeting Criteria:[/bold] {len(result.analyses)}\n"
                f"[bold]Minimum Score:[/bold] {min_score}",
                title="Scan Statistics",
                border_style="cyan",
            )
            console.print(stats_panel)
            console.print()
            
            # Market summary
            if result.market_summary:
                console.print(Panel(
                    result.market_summary,
                    title="Market Summary",
                    border_style="blue",
                ))
                console.print()
            
            # Results table
            table = Table(title=f"Top {top} Stock Picks (from {len(result.analyses)} qualifying)", box=box.ROUNDED)
            table.add_column("#", style="dim", width=3)
            table.add_column("Symbol", style="cyan")
            table.add_column("Name", max_width=20)
            table.add_column("Price", justify="right")
            table.add_column("Score", justify="right")
            table.add_column("Recommendation", justify="center")
            table.add_column("Confidence", justify="right")
            
            for i, analysis in enumerate(result.analyses[:top], 1):
                rec_color = get_recommendation_color(analysis.recommendation)
                score_color = get_score_color(analysis.total_score)
                
                table.add_row(
                    str(i),
                    analysis.symbol.replace('.TA', ''),
                    analysis.name[:20] if analysis.name else "-",
                    f"₪{analysis.current_price:.2f}",
                    f"[{score_color}]{analysis.total_score:.0f}[/]",
                    f"[{rec_color}]{analysis.recommendation.value}[/]",
                    f"{analysis.confidence:.0f}%",
                )
            
            console.print(table)
            console.print()
            
            # Show AI insights for top picks
            if result.ai_insights:
                console.print("[bold]AI Insights for Top Picks:[/bold]")
                console.print()
                
                for symbol, ai in list(result.ai_insights.items())[:3]:
                    console.print(f"[cyan]{symbol}:[/cyan] {ai.summary[:200]}...")
                    console.print()
            
        finally:
            await agent.close()
    
    asyncio.run(run_scan())


@cli.command()
@click.option('--budget', required=True, type=float, help='Investment budget in ILS')
@click.option('--risk', default='moderate', 
              type=click.Choice(['conservative', 'moderate', 'aggressive']),
              help='Risk tolerance level')
def portfolio(budget: float, risk: str):
    """Get portfolio allocation recommendations.
    
    Example: python -m src.main portfolio --budget 50000 --risk moderate
    """
    settings = get_settings()
    
    async def run_portfolio():
        agent = StockAgent(settings)
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description="Building portfolio...", total=None)
                result = await agent.build_portfolio(budget, risk)
            
            console.print()
            
            # Portfolio summary
            summary = Panel(
                f"[bold]Budget:[/bold] ₪{result.total_budget:,.0f}\n"
                f"[bold]Risk Level:[/bold] {result.risk_level.value.title()}\n"
                f"[bold]Positions:[/bold] {len(result.allocations)}\n"
                f"[bold]Cash Reserve:[/bold] ₪{result.cash_reserve:,.0f} ({result.cash_reserve_percent:.1f}%)\n"
                f"[bold]Diversification:[/bold] {result.diversification_score:.0f}/100\n"
                f"[bold]Expected Return:[/bold] {result.expected_return:.1f}%\n"
                f"[bold]Max Drawdown:[/bold] {result.max_drawdown:.1f}%",
                title="Portfolio Recommendation",
                border_style="green",
            )
            console.print(summary)
            console.print()
            
            if not result.allocations:
                console.print("[yellow]No suitable investments found for current criteria.[/yellow]")
                return
            
            # Allocations table
            table = Table(title="Recommended Allocations", box=box.ROUNDED)
            table.add_column("Symbol", style="cyan")
            table.add_column("Name", max_width=15)
            table.add_column("Shares", justify="right")
            table.add_column("Investment", justify="right")
            table.add_column("Allocation", justify="right")
            table.add_column("Entry", justify="right")
            table.add_column("Target", justify="right")
            table.add_column("Stop", justify="right")
            
            for alloc in result.allocations:
                table.add_row(
                    alloc.symbol.replace('.TA', ''),
                    alloc.name[:15] if alloc.name else "-",
                    str(alloc.shares),
                    f"₪{alloc.investment_amount:,.0f}",
                    f"{alloc.allocation_percent:.1f}%",
                    f"₪{alloc.entry_price:.2f}",
                    f"₪{alloc.target_price:.2f}" if alloc.target_price else "-",
                    f"₪{alloc.stop_loss:.2f}" if alloc.stop_loss else "-",
                )
            
            console.print(table)
            
        finally:
            await agent.close()
    
    asyncio.run(run_portfolio())


@cli.command()
@click.argument('symbols', nargs=-1)
@click.option('--interval', default=5, help='Update interval in minutes')
def watch(symbols: tuple, interval: int):
    """Monitor stocks in real-time.
    
    Example: python -m src.main watch TEVA ICL NICE --interval 5
    """
    if not symbols:
        symbols = ('TEVA', 'ICL', 'NICE')
    
    settings = get_settings()
    interval_seconds = interval * 60
    
    console.print(f"[bold]Monitoring {len(symbols)} stocks[/bold]")
    console.print(f"Update interval: {interval} minutes")
    console.print("Press Ctrl+C to stop\n")
    
    async def run_watch():
        agent = StockAgent(settings)
        
        try:
            while True:
                # Create table
                table = Table(
                    title=f"Stock Monitor - {datetime.now().strftime('%H:%M:%S')}",
                    box=box.ROUNDED,
                )
                table.add_column("Symbol", style="cyan")
                table.add_column("Price", justify="right")
                table.add_column("Score", justify="right")
                table.add_column("Recommendation", justify="center")
                table.add_column("Technical", justify="right")
                table.add_column("Fundamental", justify="right")
                table.add_column("Sentiment", justify="right")
                
                for symbol in symbols:
                    try:
                        analysis, _ = await agent.analyze_stock(symbol, include_ai=False)
                        
                        rec_color = get_recommendation_color(analysis.recommendation)
                        score_color = get_score_color(analysis.total_score)
                        
                        table.add_row(
                            symbol,
                            f"₪{analysis.current_price:.2f}",
                            f"[{score_color}]{analysis.total_score:.0f}[/]",
                            f"[{rec_color}]{analysis.recommendation.value}[/]",
                            f"{analysis.technical_score:.0f}",
                            f"{analysis.fundamental_score:.0f}",
                            f"{analysis.sentiment_score:.0f}",
                        )
                    except Exception as e:
                        table.add_row(symbol, "Error", "-", "-", "-", "-", "-")
                
                # Clear and print
                console.clear()
                console.print(table)
                console.print(f"\n[dim]Next update in {interval} minutes. Press Ctrl+C to stop.[/dim]")
                
                await asyncio.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped.[/yellow]")
        finally:
            await agent.close()
    
    asyncio.run(run_watch())


@cli.command()
def sentiment():
    """Get current market sentiment from news.
    
    Example: python -m src.main sentiment
    """
    settings = get_settings()
    
    async def run_sentiment():
        agent = StockAgent(settings)
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description="Analyzing market sentiment...", total=None)
                sentiment = await agent.get_market_sentiment()
            
            console.print()
            
            # Sentiment color
            label = sentiment['label']
            if 'positive' in label:
                color = "green"
            elif 'negative' in label:
                color = "red"
            else:
                color = "yellow"
            
            # Display panel
            panel = Panel(
                f"[bold]Overall Sentiment:[/bold] [{color}]{label.replace('_', ' ').title()}[/{color}]\n"
                f"[bold]Score:[/bold] {sentiment['overall_score']:.2f} (-1 to +1 scale)\n"
                f"[bold]Trend:[/bold] {sentiment['trend'].title()}\n"
                f"[bold]Articles Analyzed:[/bold] {sentiment['articles_analyzed']}\n"
                f"[bold]Breakdown:[/bold] {sentiment['positive']} positive, "
                f"{sentiment['negative']} negative, {sentiment['neutral']} neutral",
                title="Market Sentiment",
                border_style=color,
            )
            console.print(panel)
            
            # Headlines
            if sentiment['headlines']:
                console.print("\n[bold]Recent Headlines:[/bold]")
                for headline in sentiment['headlines']:
                    console.print(f"  • {headline}")
            
        finally:
            await agent.close()
    
    asyncio.run(run_sentiment())


@cli.command(name='list')
@click.option('--indices', is_flag=True, help='Show only TA-35/TA-125 index components')
def list_stocks(indices: bool):
    """List all available TASE stocks.
    
    Fetches the current list of tradeable stocks from the exchange.
    
    Example: python -m src.main list --indices
    """
    settings = get_settings()
    
    async def run_list():
        agent = StockAgent(settings)
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description="Fetching TASE stock list...", total=None)
                symbols = await agent.get_all_tase_symbols(use_indices=indices)
            
            console.print()
            
            # Display in columns
            table = Table(
                title=f"{'TASE Index' if indices else 'All TASE'} Stocks ({len(symbols)} total)",
                box=box.ROUNDED,
                show_lines=False,
            )
            
            # Create multiple columns
            cols = 5
            table.add_column("", style="cyan")
            table.add_column("", style="cyan")
            table.add_column("", style="cyan")
            table.add_column("", style="cyan")
            table.add_column("", style="cyan")
            
            # Fill rows
            for i in range(0, len(symbols), cols):
                row = symbols[i:i+cols]
                while len(row) < cols:
                    row.append("")
                table.add_row(*row)
            
            console.print(table)
            console.print()
            console.print(f"[dim]Use 'python -m src.main scan --all' to analyze all stocks[/dim]")
            
        finally:
            await agent.close()
    
    asyncio.run(run_list())


@cli.command()
@click.option('--min-volume', default=100000, help='Minimum average daily volume')
@click.option('--min-cap', default=100, help='Minimum market cap in millions ILS')
def discover(min_volume: int, min_cap: int):
    """Discover all tradeable TASE stocks.
    
    Fetches stocks dynamically from the TASE exchange and filters
    by liquidity and market cap criteria.
    
    Example: python -m src.main discover --min-volume 500000 --min-cap 500
    """
    settings = get_settings()
    min_cap_ils = min_cap * 1_000_000  # Convert to ILS
    
    async def run_discover():
        agent = StockAgent(settings)
        
        try:
            console.print("[bold cyan]Discovering TASE stocks...[/bold cyan]")
            console.print(f"Criteria: Volume >= {min_volume:,}, Market Cap >= ₪{min_cap}M")
            console.print()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description="Fetching and filtering stocks...", total=None)
                
                # First get all symbols
                all_symbols = await agent.get_all_tase_symbols()
                
            console.print(f"\n[bold]Found {len(all_symbols)} total TASE symbols[/bold]")
            
            # Filter by criteria
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description="Applying filters...", total=None)
                qualifying = await agent.discover_stocks(
                    min_volume=min_volume,
                    min_market_cap=min_cap_ils,
                )
            
            console.print()
            
            if qualifying:
                # Display results
                table = Table(title=f"Discovered {len(qualifying)} Tradeable Stocks", box=box.ROUNDED)
                table.add_column("#", style="dim", width=3)
                table.add_column("Symbol", style="cyan")
                
                for i, symbol in enumerate(qualifying, 1):
                    table.add_row(str(i), symbol)
                
                console.print(table)
                console.print()
                console.print(f"[dim]Run 'python -m src.main scan' to analyze these stocks[/dim]")
            else:
                console.print("[yellow]No stocks found matching the criteria.[/yellow]")
                
        finally:
            await agent.close()
    
    asyncio.run(run_discover())


@cli.command()
def config():
    """Show current configuration status."""
    settings = get_settings()
    
    console.print(Panel(
        f"[bold]Gemini API:[/bold] {'✓ Configured' if settings.gemini_api_key else '✗ Not configured'}\n"
        f"[bold]TASE API:[/bold] {'✓ Configured' if settings.tase_api_key else '✗ Not configured'}\n"
        f"\n[bold]Analysis Weights:[/bold]\n"
        f"  Technical: {settings.technical_weight * 100:.0f}%\n"
        f"  Fundamental: {settings.fundamental_weight * 100:.0f}%\n"
        f"  Sentiment: {settings.sentiment_weight * 100:.0f}%\n"
        f"\n[bold]Trading Parameters:[/bold]\n"
        f"  Min Holding: {settings.min_holding_days} days\n"
        f"  Max Holding: {settings.max_holding_days} days\n"
        f"\n[bold]Rate Limits:[/bold]\n"
        f"  Gemini: {settings.gemini_rpm} req/min\n"
        f"  TASE: {settings.tase_rpm} req/min",
        title="Configuration Status",
        border_style="blue",
    ))
    
    # Show missing keys
    missing = settings.validate_keys()
    if missing:
        console.print(f"\n[yellow]Missing API keys: {', '.join(missing)}[/yellow]")
        console.print("Copy .env.example to .env and add your API keys.")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
