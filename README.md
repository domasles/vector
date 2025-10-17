![VECTOR Logo](https://raw.githubusercontent.com/domasles/vector/main/VECTORLogo.png)

# VECTOR - VECTOR Encodes Coordinates To Optimize Retrieval

A lightweight vector database library for Python that organizes data using mathematical coordinate systems. Built with domain-driven architecture and designed for single-file storage with O(1) lookup performance.

## Project Philosophy

Vector embraces the "coordinate-based data organization" approach where every table must have an X-axis as the primary key, with other attributes representing relationships between dimensions. This creates a natural mathematical model for data organization:

- **X-axis (Central Axis)**: Primary key and coordinate system foundation
- **Y, Z, J... (Dimensional Spaces)**: Additional attributes that define relationships
- **Coordinate Mappings**: Functions that map between dimensional spaces
- **Vector Points**: Individual data records positioned in the coordinate space

## Key Features

### Vector Mathematics Foundation
- **Coordinate System Architecture**: Data organized around mathematical coordinate principles
- **Dimensional Spaces**: Scalable N-dimensional data representation without structural changes
- **O(1) Lookup Performance**: Coordinate indexing for instant data retrieval
- **Value Deduplication**: Automatic optimization of storage through value deduplication in dimensional spaces

### Domain-Driven Architecture
- **Clean Architecture**: Separation of domain logic, application services, and infrastructure
- **Coordinate Abstractions**: Rich domain objects representing mathematical concepts
- **Immutable Value Objects**: Thread-safe coordinate and mapping representations
- **Repository Patterns**: Clean data access interfaces

## Quick Start

### Installation

```python
# Clone the repository
git clone <repository-url>
cd vector

# Install in development mode
pip install -e .

# Or install from PyPI (when published)
pip install vector-datalib
```

### Basic Usage

```python
from vector_db import VectorDB

# Create or open a database
db = VectorDB("my_data.db")

# Insert data (X-axis is always the primary key)
db.insert(x_coordinate=1, age=25, name="Alice")
db.insert(x_coordinate=2, age=30, name="Bob")
db.insert(x_coordinate=3, age=25, name="Charlie")  # age=25 deduplicated automatically

# O(1) coordinate-based lookup
person = db.lookup(x_coordinate=1)
print(person)  # {'age': 25, 'name': 'Alice'}

# Insert additional dimensions
db.insert(x_coordinate=4, age=28, name="Diana", city="New York")

# Update existing coordinates
db.update(x_coordinate=1, age=26)  # Alice's age updated

# Save to persistent storage
db.save("my_data.db")

# Load from storage
db2 = VectorDB.load("my_data.db")
```

### Coordinate System Concepts

```python
# X-axis serves as the central coordinate system
# Other dimensions (Y, Z, J...) represent relationships

# Example: User data with coordinate-based organization
db = VectorDB("users.db")

# X-coordinate: User ID (primary key)
# Y-dimension: Age (relationship to time)
# Z-dimension: Name (relationship to identity)
db.insert(x_coordinate=101, age=25, name="Alice")
db.insert(x_coordinate=102, age=30, name="Bob")

# Dimensional expansion - add new coordinate without schema changes
db.insert(x_coordinate=103, age=25, name="Charlie", department="Engineering")
```

## Architecture

Vector follows clean architecture principles with mathematical domain modeling:

```
src/vector_db/
├── domain/
│   ├── coordinates/            # X-axis coordinate system (primary key)
│   ├── spaces/                 # Y, Z, J... dimensional spaces  
│   ├── mappings/               # Functions between dimensional spaces
│   └── __init__.py
├── application/
│   ├── main.py                 # Main database API
│   └── __init__.py
├── infrastructure/
│   ├── storage/                # .db file persistence
│   └── __init__.py
├── meta.py                     # Version and metadata
└── __init__.py                 # Public API exports
```

### Domain Layer

- **CentralAxis**: Manages X-coordinate system and primary key constraints
- **DimensionalSpace**: Handles Y, Z, J... dimensions with value deduplication
- **CoordinateMapping**: Maps relationships between dimensional spaces
- **VectorPoint**: Represents individual data records as coordinate positions

### Application Layer

- **VectorDB**: Main database interface providing the scripting API
- **Coordinate Operations**: Insert, lookup, update operations on coordinate system
- **Dimensional Management**: Dynamic expansion and contraction of coordinate spaces

### Infrastructure Layer

- **VectorFileStorage**: Handles .db file format with JSON and gzip compression
- **Persistence Management**: Atomic save/load operations with metadata

## Mathematical Model

### Coordinate System Design

All tables in Vector must follow the coordinate system principle:

- **X-axis (Primary Key)**: Central coordinate that uniquely identifies each vector point
- **Dimensional Relationships**: Other attributes represent relationships between the X-coordinate and various dimensional spaces

```python
# Mathematical representation:
# Point P at coordinate X has relationships to multiple dimensions
# P(x) = {Y: f_y(x), Z: f_z(x), J: f_j(x), ...}
# where f_axis represents the mapping function for each dimension

db.insert(x_coordinate=1, age=25, name="Alice", city="Boston")
# Creates: P(1) = {age: f_age(1)=25, name: f_name(1)="Alice", city: f_city(1)="Boston"}
```

### Value Deduplication

Vector automatically optimizes storage by deduplicating values within dimensional spaces:

```python
db.insert(x_coordinate=1, age=25, name="Alice")
db.insert(x_coordinate=2, age=25, name="Bob")    # age=25 stored once
db.insert(x_coordinate=3, age=25, name="Charlie") # age=25 referenced

# Storage optimization: age=25 stored once, referenced by multiple coordinates
```

### N-Dimensional Scalability

Add new dimensions without structural changes:

```python
# Start with 2 dimensions
db.insert(x_coordinate=1, age=25, name="Alice")

# Expand to 3 dimensions
db.insert(x_coordinate=2, age=30, name="Bob", city="Boston")

# Expand to N dimensions dynamically
db.insert(x_coordinate=3, age=25, name="Charlie", city="Boston", department="Engineering")
```

## Performance Characteristics

### Time Complexity
- **Insert**: O(1) average case with hash-based coordinate indexing
- **Lookup**: O(1) direct coordinate access
- **Update**: O(1) coordinate-based modification
- **Dimensional Expansion**: O(1) addition of new coordinate relationships

### Space Complexity
- **Value Deduplication**: Automatic optimization reduces memory usage
- **Coordinate Indexing**: Hash-based storage for constant-time access
- **Compression**: Gzip compression for persistent storage efficiency

## File Format

### .db File Structure

```json
{
  "metadata": {
    "version": "1.0.1-beta",
    "created_at": "2025-01-XX",
    "coordinate_count": 1000
  },
  "central_axis": {
    "coordinates": [1, 2, 3, ...]
  },
  "dimensional_spaces": {
    "age": {
      "values": [25, 30, 35],
      "coordinate_mappings": {"1": 0, "2": 1, "3": 0}
    },
    "name": {
      "values": ["Alice", "Bob", "Charlie"],
      "coordinate_mappings": {"1": 0, "2": 1, "3": 2}
    }
  }
}
```

## Development

### Requirements
- Python 3.7+
- No external dependencies (uses only standard library)

## Coordinate System Examples

### User Management System

```python
db = VectorDB("users.db")

# X-coordinate: User ID, Y-dimension: Profile data
db.insert(x_coordinate=1001, name="Alice Johnson", age=28, department="Engineering")
db.insert(x_coordinate=1002, name="Bob Smith", age=32, department="Sales")
db.insert(x_coordinate=1003, name="Charlie Brown", age=28, department="Engineering")

# O(1) user lookup
user = db.lookup(x_coordinate=1001)
print(f"User: {user['name']}, Age: {user['age']}")

# Dynamic expansion - add new dimensional relationships
db.update(x_coordinate=1001, salary=75000, location="Boston")
```

### Product Catalog

```python
db = VectorDB("products.db")

# X-coordinate: Product ID, Y/Z dimensions: Product attributes
db.insert(x_coordinate=2001, name="Laptop", price=999.99, category="Electronics")
db.insert(x_coordinate=2002, name="Mouse", price=29.99, category="Electronics")
db.insert(x_coordinate=2003, name="Desk", price=299.99, category="Furniture")

# Value deduplication automatically optimizes "Electronics" category storage
```

## Best Practices

### Coordinate System Design
- **Always use X-axis as primary key**: This maintains the mathematical foundation
- **Design dimensional relationships**: Think about how attributes relate to coordinates
- **Leverage value deduplication**: Repeated values in dimensions are automatically optimized
- **Plan for dimensional expansion**: Design coordinate spaces that can grow dynamically

### Performance Optimization
- **Use appropriate coordinate ranges**: Choose coordinate values that distribute well
- **Batch operations**: Group multiple insertions when possible
- **Regular persistence**: Save to .db files at appropriate intervals
- **Monitor dimensional growth**: Large numbers of unique values reduce deduplication benefits

### Data Organization
- **Logical coordinate grouping**: Group related data with nearby coordinates when possible
- **Consistent dimensional naming**: Use clear, consistent names for dimensional spaces
- **Document coordinate meanings**: Maintain documentation of what each coordinate represents

## Troubleshooting

### Common Issues

**Large file sizes with compressed storage**:
- Check for high dimensional diversity (many unique values)
- Consider coordinate space reorganization for better deduplication

**Performance degradation**:
- Monitor the number of unique values in dimensional spaces
- Consider splitting large coordinate spaces into multiple databases

## Contributing

1. Fork the repository
2. Create a feature branch following the coordinate system principles
3. Implement changes with proper domain modeling
4. Ensure mathematical consistency in coordinate operations
5. Submit a pull request

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Vector Mathematics

Vector database design is inspired by mathematical vector spaces where:

- **Coordinates define position**: X-axis establishes the coordinate system foundation
- **Dimensions represent relationships**: Each dimension shows how data relates to coordinates
- **Mappings preserve structure**: Functions between dimensions maintain mathematical consistency
- **Scalability through expansion**: N-dimensional growth without architectural changes

The name "Vector" reflects this mathematical foundation where data points exist as vectors in a coordinate space, with the X-axis serving as the primary coordinate system and other dimensions representing the vector's components in different spaces.

---

**Organize your data with mathematical precision. Scale with coordinate clarity.**

*Built for developers who appreciate clean architecture and mathematical elegance.*
