"""
Pytest configuration and shared fixtures for SQL Agent tests.
"""

import pytest
import sqlite3
import tempfile
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_db() -> str:
    """
    Create a sample SQLite database for testing.

    Returns:
        Path to the temporary database file.
    """
    db_fd, db_path = tempfile.mkstemp(suffix='.db')

    # Create connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create sample tables
    cursor.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            created_at DATE
        )
    ''')

    cursor.execute('''
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            amount DECIMAL(10, 2),
            status TEXT,
            created_at DATE,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # Insert sample data
    users_data = [
        (1, 'Alice Johnson', 'alice@example.com', '2023-01-15'),
        (2, 'Bob Smith', 'bob@example.com', '2023-02-20'),
        (3, 'Charlie Brown', 'charlie@example.com', '2023-03-10'),
        (4, 'Diana Prince', 'diana@example.com', '2023-04-05'),
        (5, 'Eve Wilson', 'eve@example.com', '2023-05-12'),
    ]

    cursor.executemany(
        'INSERT INTO users (id, name, email, created_at) VALUES (?, ?, ?, ?)',
        users_data
    )

    orders_data = [
        (1, 1, 100.50, 'completed', '2023-06-01'),
        (2, 1, 250.00, 'completed', '2023-06-15'),
        (3, 2, 75.25, 'pending', '2023-06-20'),
        (4, 3, 500.00, 'completed', '2023-07-01'),
        (5, 2, 125.75, 'completed', '2023-07-10'),
        (6, 4, 300.00, 'pending', '2023-07-15'),
        (7, 5, 450.00, 'completed', '2023-07-20'),
    ]

    cursor.executemany(
        'INSERT INTO orders (id, user_id, amount, status, created_at) VALUES (?, ?, ?, ?, ?)',
        orders_data
    )

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    Path(db_path).unlink()


@pytest.fixture
def sample_csv(tmp_path) -> str:
    """
    Create a sample CSV file for testing.

    Args:
        tmp_path: Pytest tmp_path fixture.

    Returns:
        Path to the CSV file.
    """
    csv_file = tmp_path / "sample_data.csv"

    df = pd.DataFrame({
        'product': ['A', 'B', 'C', 'D', 'E'],
        'sales': [100, 250, 150, 300, 200],
        'region': ['North', 'South', 'East', 'West', 'North'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    })

    df.to_csv(csv_file, index=False)

    return str(csv_file)


@pytest.fixture
def sample_json(tmp_path) -> str:
    """
    Create a sample JSON file for testing.

    Args:
        tmp_path: Pytest tmp_path fixture.

    Returns:
        Path to the JSON file.
    """
    json_file = tmp_path / "sample_data.json"

    import json
    data = [
        {'id': 1, 'name': 'Product A', 'price': 29.99},
        {'id': 2, 'name': 'Product B', 'price': 49.99},
        {'id': 3, 'name': 'Product C', 'price': 19.99},
    ]

    with open(json_file, 'w') as f:
        json.dump(data, f)

    return str(json_file)


@pytest.fixture
def sample_data() -> dict:
    """
    Provide sample data for testing.

    Returns:
        Dictionary with sample data.
    """
    return {
        'sql_examples': [
            {
                'sql': 'SELECT * FROM users WHERE created_at > "2023-01-01"',
                'expected_rows': 5,
            },
            {
                'sql': 'SELECT COUNT(*) as count FROM orders WHERE status = "completed"',
                'expected_rows': 1,
            },
            {
                'sql': 'SELECT user_id, SUM(amount) as total FROM orders GROUP BY user_id',
                'expected_rows': 5,
            },
        ],
        'chart_configs': [
            {
                'type': 'bar',
                'x_axis': 'product',
                'y_axis': 'sales',
                'title': 'Sales by Product',
            },
            {
                'type': 'line',
                'x_axis': 'date',
                'y_axis': 'amount',
                'title': 'Amount Trend',
            },
            {
                'type': 'pie',
                'labels': 'region',
                'values': 'sales',
                'title': 'Sales by Region',
            },
        ],
    }
