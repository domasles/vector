![VECTOR Logo](https://raw.githubusercontent.com/domasles/vector/main/VECTORLogo.png)

[![Python code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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

# Or install from PyPI
pip install vector-datalib
```

### Basic Usage

**Async-First API with Context Managers**

```python
import asyncio
from vector_datalib import VectorDB

async def main():
    # Create database with async context manager
    async with VectorDB("my_data.db") as db:
        # 1. UPSERT: Insert or update (one method for both!)
        await db.upsert(101, {"age": 25, "name": "Alice"})
        await db.upsert(102, {"age": 30, "name": "Bob"})
        await db.upsert(101, {"age": 26, "name": "Alice"})  # Updates Alice

        # 2. LOOKUP: O(1) coordinate-based retrieval
        name = await db.lookup(101, "name")
        print(f"Employee 101: {name}")  # Employee 101: Alice

        # 3. DELETE: Remove records with automatic cleanup
        await db.delete(102)  # Bob is gone, values cleaned up

        # Batch operations for concurrency
        await db.batch_upsert([
            (104, {"name": "Diana", "age": 28}),
            (105, {"name": "Eve", "age": 32}),
            (106, {"name": "Frank", "age": 27})
        ])

        # Database automatically saved on exit

asyncio.run(main())
```

**The Entire API:**
- `await upsert(key, attributes)` - Insert or update
- `await lookup(key, dimension)` - Read data (O(1))
- `await delete(key)` - Remove with cleanup
- `await batch_upsert(records)` - Concurrent insert/update
- `await batch_lookup(queries)` - Concurrent reads
- `await batch_delete(keys)` - Concurrent deletes
- `await save()` - Manual save (auto-saves on exit)
- `await get_stats()` - Database statistics

### Advanced Patterns

```python
import asyncio
from vector_datalib import VectorDB

async def main():
    async with VectorDB("analytics.db") as db:
        # Concurrent batch upserts
        records = [(i, {"value": i, "squared": i**2}) for i in range(1000)]
        coordinates = await db.batch_upsert(records)
        print(f"Inserted {len(coordinates)} records concurrently")

        # Batch lookups (all concurrent)
        user_queries = [(101, "name"), (102, "age"), (103, "department")]
        results = await db.batch_lookup(user_queries)

        # LRU caching automatically optimizes repeated lookups
        name1 = await db.lookup(101, "name")  # Database + cache
        name2 = await db.lookup(101, "name")  # Cache hit (instant)

        # Batch deletes with cleanup
        to_delete = [i for i in range(0, 1000, 2)]  # Delete evens
        deleted_count = await db.batch_delete(to_delete)
        print(f"Deleted {deleted_count} records")

        # Manual save (optional - auto-saves on context exit)
        await db.save()

        # Database statistics
        stats = await db.get_stats()
        print(f"Database: {stats['vector_points']} points, {stats['dimensions']} dimensions")

asyncio.run(main())
```

**Concurrency Benefits:**
- `batch_upsert()`: All upserts execute concurrently using `asyncio.gather()`
- `batch_lookup()`: All lookups execute concurrently
- `batch_delete()`: All deletes execute concurrently with automatic value cleanup
- Cache-safe: `asyncio.Lock` prevents race conditions
- Non-blocking I/O: Uses `aiofiles` for async file operations

## Architecture

Vector follows clean architecture principles with mathematical domain modeling:

```
src/vector_datalib/
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

import asyncio
from vector_datalib import VectorDB

async def main():
    async with VectorDB("data.db") as db:
        await db.upsert(101, age=25, name="Alice", city="Boston")
        # Creates: P(101) = {age: f_age(101)=25, name: f_name(101)="Alice", city: f_city(101)="Boston"}
        
        # Async lookup
        record = await db.lookup(101)
        print(record)  # {"age": 25, "name": "Alice", "city": "Boston"}

asyncio.run(main())
```

### Value Deduplication

Vector automatically optimizes storage by deduplicating values within dimensional spaces:

```python
import asyncio
from vector_datalib import VectorDB

async def main():
    async with VectorDB("data.db") as db:
        await db.upsert(101, age=25, name="Alice")
        await db.upsert(102, age=25, name="Bob")     # age=25 stored once
        await db.upsert(103, age=25, name="Charlie") # age=25 referenced

        # Storage optimization: age=25 stored once, referenced by multiple coordinates
        
        # All lookups return the correct values
        records = await db.batch_lookup([101, 102, 103])
        print(f"Records: {records}")  # All have age=25, but stored once internally

asyncio.run(main())
```

### N-Dimensional Scalability

Add new dimensions without structural changes:

```python
import asyncio
from vector_datalib import VectorDB

async def main():
    async with VectorDB("data.db") as db:
        # Start with 2 dimensions
        await db.upsert(101, age=25, name="Alice")

        # Expand to 3 dimensions (just upsert with new fields)
        await db.upsert(102, age=30, name="Bob", city="Boston")

        # Expand to N dimensions dynamically
        await db.upsert(103, age=25, name="Charlie", city="Boston", department="Engineering")
        
        # Query across dimensions
        charlie = await db.lookup(103)
        print(f"Charlie: {charlie}")

asyncio.run(main())
```

## Performance Characteristics

### Time Complexity
- **Upsert**: O(1) average case with hash-based coordinate indexing (async)
- **Lookup**: O(1) direct coordinate access + cache check (async)
- **Delete**: O(1) tombstoning with no coordinate shifting (async)
- **Dimensional Expansion**: O(1) addition of new coordinate relationships
- **Batch Operations**: O(n) with concurrent execution via `asyncio.gather()`

### Storage Optimizations
- **MessagePack Serialization**: 2-3x smaller files than JSON
- **LZ4 Compression**: Blazing fast compression
- **Async I/O**: Non-blocking file operations with `aiofiles`
- **LRU Caching**: In-memory caching for frequently accessed data with `asyncio.Lock`
- **Concurrent Safety**: `asyncio.Lock` prevents race conditions in cache and storage
- **Tombstoning**: O(1) deletion without coordinate shifting overhead
- **Tombstone Slot Reuse**: Deleted coordinate slots are recycled for new inserts
- **Reference Counting**: Automatic cleanup of unreferenced values
- **Context Managers**: Automatic resource management and cleanup (async with `__aenter__`/`__aexit__`)

### Space Complexity
- **Value Deduplication**: Automatic optimization reduces memory usage
- **Coordinate Indexing**: Hash-based storage for constant-time access
- **LZ4 Compression**: Fast compression for persistent storage efficiency
- **Sparse Storage**: Tombstones minimize wasted space

## Async-First Architecture

### Design Philosophy

Vector uses a **simplified async-first** API design:

- **All I/O operations are async**:
  - `upsert()` - Insert or update (single method for both)
  - `lookup()` - Cache + coordinate retrieval
  - `delete()` - Tombstoning with automatic cleanup
  - `save()` / `load()` - File persistence
  - `batch_upsert()` / `batch_lookup()` / `batch_delete()` - Concurrent operations
  - `get_stats()` - Database statistics

- **No insert() vs update() confusion**: Only `upsert()` for writes
- **No verification needed**: Internal consistency maintained automatically
- **Batching in facade**: Concurrent operations handled by main API

### Concurrency Features

- **Non-blocking I/O**: `aiofiles` for async file operations
- **Concurrent batching**: `asyncio.gather()` for parallel operations
- **Cache safety**: `asyncio.Lock` prevents race conditions
- **No blocking locks**: Removed `threading.RLock` and `filelock`
- **Tombstoning**: O(1) deletion without coordinate shifting

### Migration from Older API

If migrating from older mixed sync/async API:

```python
# Old (mixed sync/async):
with VectorDB("data.db") as db:
    db.insert(101, {"name": "Alice", "age": 25})  # Sync
    name = await db.lookup(101)  # Async
    await db.update(101, age=26)  # Async

# New (async-first with upsert):
async with VectorDB("data.db") as db:
    await db.upsert(101, name="Alice", age=25)  # Async upsert
    name = await db.lookup(101)  # Async lookup
    await db.upsert(101, name="Alice", age=26)  # Async upsert (update)
```

## File Format

### .db File Structure

Vector uses a binary file format with MessagePack serialization and LZ4 compression for optimal performance:

```
┌─────────────────────────────────────┐
│           LZ4 Compressed            │
│  ┌───────────────────────────────┐  │
│  │      MessagePack Binary       │  │
│  │  ┌─────────────────────────┐  │  │
│  │  │ metadata                │  │  │
│  │  │ central_axis            │  │  │
│  │  │ dimensional_spaces      │  │  │
│  │  │ coordinate_mappings     │  │  │
│  │  └─────────────────────────┘  │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

**Structure contents:**
- `metadata`: Version, timestamps, statistics
- `central_axis`: Vector points, coordinate map, free slots for reuse
- `dimensional_spaces`: Value domains with deduplication
- `coordinate_mappings`: Coordinate-to-value-id mappings with reference counts

## Development

### Requirements
- Python 3.11+
- Dependencies: msgpack, aiofiles, lz4, bidict

## Coordinate System Examples

### User Management System

```python
import asyncio
from vector_datalib import VectorDB

async def main():
    async with VectorDB("users.db") as db:
        # X-coordinate: User ID, Y-dimension: Profile data
        await db.upsert(1001, name="Alice Johnson", age=28, department="Engineering")
        await db.upsert(1002, name="Bob Smith", age=32, department="Sales")  
        await db.upsert(1003, name="Charlie Brown", age=28, department="Engineering")

        # O(1) user lookup (async)
        user = await db.lookup(1001)
        print(f"User: {user}")

        # Update using upsert - add new dimensional relationships
        await db.upsert(1001, name="Alice Johnson", age=28, department="Engineering",
                       salary=75000, location="Boston")

asyncio.run(main())
```

### Product Catalog

```python
import asyncio
from vector_datalib import VectorDB

async def main():
    async with VectorDB("products.db") as db:
        # X-coordinate: Product ID, Y/Z dimensions: Product attributes
        await db.upsert(2001, name="Laptop", price=999.99, category="Electronics")
        await db.upsert(2002, name="Mouse", price=29.99, category="Electronics")
        await db.upsert(2003, name="Desk", price=299.99, category="Furniture")

        # Value deduplication automatically optimizes "Electronics" category storage

        # Concurrent batch lookup
        products = await db.batch_lookup([2001, 2002, 2003])
        print(f"Products: {products}")

asyncio.run(main())
```

## Best Practices

### Simplified API Usage
- **One write method**: `await db.upsert()` handles both insert and update
- **No insert() vs update() confusion**: upsert does it all
- **Automatic cleanup**: delete() removes unused values automatically
- **Tombstoning**: O(1) deletion with no coordinate shifting
- **Always use async context managers**: `async with VectorDB() as db:`
- **Batch for concurrency**: Use batch_* methods for multiple operations
- **Async everywhere**: All I/O operations are async for consistency

### Coordinate System Design
- **Always use X-axis as primary key**: This maintains the mathematical foundation
- **Design dimensional relationships**: Think about how attributes relate to coordinates
- **Leverage value deduplication**: Repeated values in dimensions are automatically optimized
- **Plan for dimensional expansion**: Design coordinate spaces that can grow dynamically

### Performance Optimization
- **Leverage batch operations**: Use concurrent batching for multiple operations
- **LRU cache awareness**: Repeated lookups are cached automatically
- **Appropriate coordinate ranges**: Choose coordinate values that distribute well
- **Monitor dimensional growth**: Large numbers of unique values reduce deduplication benefits
- **Use asyncio best practices**: Don't block the event loop in your code

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
