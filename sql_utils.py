from contextlib import closing


def row_exists(conn, value, table_name="mol_pairs", pk_name="_id"):
    # Connect to the SQLite database
    with closing(conn.cursor()) as cursor:
        # Prepare the SQL query
        sql = f"SELECT EXISTS(SELECT 1 FROM {table_name} WHERE {pk_name} = ? LIMIT 1)"

        # Execute the query
        cursor.execute(sql, (value,))
        result = cursor.fetchone()[0]

        # Return True if the row exists, False otherwise
        return result == 1


def find_non_existing_ids(conn, ids, table_name="mol_pairs", pk_name="_id"):
    # Connect to the SQLite database
    with closing(conn.cursor()) as cursor:

        # Prepare the SQL query to check for existence
        placeholders = ', '.join('?' for _ in ids)
        sql = f"SELECT {pk_name} FROM {table_name} WHERE {pk_name} IN ({placeholders})"

        # Execute the query
        cursor.execute(sql, ids)
        existing_ids = set(row[0] for row in cursor.fetchall())

        # Find IDs that do not exist in the table
        non_existing_ids = [id_ for id_ in ids if id_ not in existing_ids]

        return non_existing_ids


def table_exists(conn, table_name):
    """Check if a table exists in the SQLite database."""
    with closing(conn.cursor()) as cursor:
        query = f"SELECT name FROM sqlite_master WHERE type='table' AND name=?;"
        cursor.execute(query, (table_name,))
        return cursor.fetchone() is not None


def create_text_table(conn, table_name, txt_columns=[], float_columns=[]):
    """Create text table with _id as the primary key."""
    with closing(conn.cursor()) as cursor:
        if cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,)):
            drop_table_sql = f"DROP TABLE IF EXISTS {table_name};"
            cursor.execute(drop_table_sql)
            print(f"Dropped existing table {table_name}")

        create_table_sql = """
        CREATE TABLE {} (
                _id TEXT PRIMARY KEY,
                {}
        );
        """.format(table_name,
                   ", \n\t\t".join([f"{c} TEXT" for c in txt_columns] + [f"{c} FLOAT" for c in float_columns]))

        cursor.execute(create_table_sql)
        conn.commit()