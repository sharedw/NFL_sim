import duckdb


class Quack:

    @staticmethod
    def fetch_table(table_name: str, db_path: str = "data/nfl.duckdb"):
        con = duckdb.connect(db_path)
        try:
            df = con.execute(f"SELECT * FROM {table_name}").fetch_df()
        finally:
            con.close()
        return df

    @staticmethod
    def query(query):
        con = duckdb.connect("data/nfl.duckdb")
        try:
            df = con.execute(query).fetch_df()
        finally:
            con.close()
        return df
    
    @staticmethod
    def file_query(query_file_name):
        con = duckdb.connect("data/nfl.duckdb")
        try:
            with open(f"./data/queries/{query_file_name}.sql", "r") as f:
                sql = f.read()
            df = con.execute(sql).fetchdf()
        finally:
            con.close()
        return df
    
    def select_columns(cols:list, alias:str=None, coalesce=False):

        if alias:
            temp = (f'{alias}.' + x for x in cols)
        else:
            temp = cols
        if coalesce:
            temp = (f'coalesce({x}, 0) as {y}' for x,y in zip(temp, cols))
        return ', '.join(temp)