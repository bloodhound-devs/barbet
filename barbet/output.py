from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl


def print_polars_df(df: "pl.DataFrame", max_rows: int = 20, column_names:list[str]|None=None):
    from rich.table import Table
    from rich.console import Console

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")

    # Add columns
    for col in column_names or df.columns:
        table.add_column(col)

    # Add rows
    for row in df.head(max_rows).iter_rows():
        table.add_row(*[str(cell) for cell in row])

    # Indicate truncation if needed
    if df.height > max_rows:
        table.add_row(*["..." for _ in df.columns])
        console.print(f"[dim]Showing first {max_rows} of {df.height} rows[/dim]")

    console.print(table)
