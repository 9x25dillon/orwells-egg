def simple_prefix_sql(q: str, table: str = "corpus", col: str = "text", k: int = 50) -> str:
    prefix = (q or "").strip().split()[0].replace("'", "''") if q else ""
    return f"SELECT * FROM {table} WHERE {col} ILIKE '{prefix}%' LIMIT {k};"