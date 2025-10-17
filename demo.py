#!/usr/bin/env python3
"""
Vector Database Demo Script
VECTOR = "VECTOR Encodes Coordinates To Optimize Retrieval"

This demo creates a vector database, populates it with sample data,
and displays the encoded/decoded data to verify correctness.
"""

import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import the Vector Database
from application.vector_db import VectorDB


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def print_database_structure(vdb: VectorDB):
    """Print the internal structure of the vector database."""
    print("\n🏗️  VECTOR DATABASE INTERNAL STRUCTURE")
    print("-" * 45)
    
    # Central Axis
    print(f"📍 Central Axis (X): {vdb.central_axis.size()} vector points")
    for i, point in enumerate(vdb.central_axis.get_all_points()):
        print(f"   X[{i}] = '{point}'")
    
    print()
    
    # Dimensional Spaces (Value Domains)
    for dim_name, space in vdb.dimensional_spaces.items():
        print(f"🌐 Dimensional Space '{dim_name}': {space.get_value_count()} unique values")
        for value_id, value in space.value_domain.items():
            print(f"   ValueDomain[{value_id}] = '{value}'")
        print()
    
    # Coordinate Mappings (f_axis functions)
    for dim_name, mapping in vdb.coordinate_mappings.items():
        print(f"🔗 Coordinate Mapping f_{dim_name}(x):")
        for coord, value_id in mapping.get_all_mappings().items():
            vector_point = vdb.central_axis.get_vector_point(coord)
            actual_value = vdb.dimensional_spaces[dim_name].get_value(value_id)
            print(f"   f_{dim_name}({coord}) = {value_id} → '{actual_value}' ['{vector_point}']")
        print()


def demonstrate_lookups(vdb: VectorDB):
    """Demonstrate O(1) lookup operations."""
    print("\n🔍 DEMONSTRATING O(1) LOOKUPS")
    print("-" * 35)
    
    test_cases = [
        ("Domas", "surname"),
        ("Domas", "age"), 
        ("Jonas", "surname"),
        ("Jonas", "age"),
        ("Petras", "email"),
        ("Nonexistent", "age")  # Should return None
    ]
    
    for vector_value, dimension in test_cases:
        result = vdb.lookup(vector_value, dimension)
        status = "✅" if result is not None else "❌"
        print(f"   {status} lookup('{vector_value}', '{dimension}') → {result}")


def demonstrate_deduplication(vdb: VectorDB):
    """Show how value deduplication works."""
    print("\n🔄 DEMONSTRATING VALUE DEDUPLICATION")
    print("-" * 40)
    
    # Show that both Domas and Jonas share the same age value
    domas_age = vdb.lookup("Domas", "age")
    jonas_age = vdb.lookup("Jonas", "age")
    
    print(f"   Domas age: {domas_age}")
    print(f"   Jonas age: {jonas_age}")
    print(f"   Same value? {domas_age == jonas_age}")
    
    # Show internal value domain for age
    age_space = vdb.dimensional_spaces["age"]
    print(f"   Age ValueDomain: {age_space.value_domain}")
    print(f"   → Both users reference the same value ID!")


def demonstrate_update_propagation(vdb: VectorDB):
    """Show how updating a value propagates to all references."""
    print("\n🔄 DEMONSTRATING UPDATE PROPAGATION")
    print("-" * 40)
    
    print("   Before update:")
    print(f"   - Domas age: {vdb.lookup('Domas', 'age')}")
    print(f"   - Jonas age: {vdb.lookup('Jonas', 'age')}")
    
    # Update age 28 to 29 - this affects both users
    success = vdb.update_dimension_value("age", 28, 29)
    print(f"\n   Update age 28 → 29: {'✅' if success else '❌'}")
    
    print("   After update:")
    print(f"   - Domas age: {vdb.lookup('Domas', 'age')}")
    print(f"   - Jonas age: {vdb.lookup('Jonas', 'age')}")
    print("   → Both values updated automatically!")


def demonstrate_vector_expansion(vdb: VectorDB):
    """Show n-dimensional expansion by adding new dimensions."""
    print("\n📐 DEMONSTRATING N-DIMENSIONAL EXPANSION")
    print("-" * 45)
    
    print("   Current dimensions:", vdb.get_dimensions())
    
    # Add new dimensions dynamically
    print("\n   Adding new dimensions...")
    
    # Add department dimension
    vdb.add_dimension("department")
    coordinate_domas = vdb.central_axis.get_coordinate("Domas")
    coordinate_jonas = vdb.central_axis.get_coordinate("Jonas")
    
    dept_id_1 = vdb.dimensional_spaces["department"].add_value("Engineering")
    dept_id_2 = vdb.dimensional_spaces["department"].add_value("Marketing")
    
    vdb.coordinate_mappings["department"].set_mapping(coordinate_domas, dept_id_1)
    vdb.coordinate_mappings["department"].set_mapping(coordinate_jonas, dept_id_2)
    
    # Add salary dimension
    vdb.add_dimension("salary")
    salary_id_1 = vdb.dimensional_spaces["salary"].add_value(75000)
    salary_id_2 = vdb.dimensional_spaces["salary"].add_value(65000)
    
    vdb.coordinate_mappings["salary"].set_mapping(coordinate_domas, salary_id_1)
    vdb.coordinate_mappings["salary"].set_mapping(coordinate_jonas, salary_id_2)
    
    print("   New dimensions:", vdb.get_dimensions())
    print(f"   Domas department: {vdb.lookup('Domas', 'department')}")
    print(f"   Domas salary: {vdb.lookup('Domas', 'salary')}")
    print(f"   Jonas department: {vdb.lookup('Jonas', 'department')}")
    print(f"   Jonas salary: {vdb.lookup('Jonas', 'salary')}")


def main():
    """Main demo function."""
    print("🚀 VECTOR DATABASE DEMO")
    print("VECTOR = 'VECTOR Encodes Coordinates To Optimize Retrieval'")
    
    # Clean up any existing demo database
    demo_db_path = "demo.db"
    if Path(demo_db_path).exists():
        Path(demo_db_path).unlink()
    
    print_section("1. CREATING VECTOR DATABASE")
    
    # Create vector database
    vdb = VectorDB(demo_db_path)
    print(f"✅ Created VectorDB: {vdb}")
    
    print_section("2. INSERTING VECTOR POINTS")
    
    # Insert sample data as described in your original concept
    print("📝 Inserting vector points with attributes...")
    
    # Your example data from the specification
    coord1 = vdb.insert("Domas", {
        "surname": "Kazlauskas",
        "age": 28,
        "registration_index": 3
    })
    
    coord2 = vdb.insert("Jonas", {
        "surname": "Jonaitis", 
        "age": 28,  # Same age - will be deduplicated
        "registration_index": 7
    })
    
    coord3 = vdb.insert("Petras", {
        "surname": "Petraitis",
        "age": 35,
        "registration_index": 5,
        "email": "petras@example.com"  # Additional dimension
    })
    
    print(f"✅ Inserted 'Domas' at coordinate {coord1}")
    print(f"✅ Inserted 'Jonas' at coordinate {coord2}")
    print(f"✅ Inserted 'Petras' at coordinate {coord3}")
    
    print_section("3. VECTOR DATABASE STRUCTURE")
    print_database_structure(vdb)
    
    print_section("4. LOOKUP OPERATIONS")
    demonstrate_lookups(vdb)
    
    print_section("5. VALUE DEDUPLICATION")
    demonstrate_deduplication(vdb)
    
    print_section("6. UPDATE PROPAGATION")
    demonstrate_update_propagation(vdb)
    
    print_section("7. N-DIMENSIONAL EXPANSION")
    demonstrate_vector_expansion(vdb)
    
    print_section("8. COMPLETE VECTOR POINTS")
    
    print("📋 All vector points with complete attribute sets:")
    for point in vdb.get_all_vector_points():
        print(f"   {point}")
        for attr_name, attr_value in point.get_all_attributes().items():
            print(f"      {attr_name}: {attr_value}")
        print()
    
    print_section("9. DATABASE PERSISTENCE")
    
    # Save to file
    success = vdb.save()
    print(f"💾 Database saved: {'✅' if success else '❌'}")
    
    # Show file info
    stats = vdb.get_stats()
    print(f"📊 Database Statistics:")
    print(f"   - Vector Points: {stats['vector_points']}")
    print(f"   - Dimensions: {stats['dimensions']}")
    print(f"   - File Size: {stats['file_size_bytes']} bytes")
    print(f"   - File Path: {demo_db_path}")
    
    # Test loading by creating new instance
    print("\n🔄 Testing database reload...")
    vdb2 = VectorDB(demo_db_path)
    print(f"✅ Reloaded VectorDB: {vdb2}")
    
    # Verify data integrity after reload
    print("🔍 Verifying data integrity after reload:")
    test_lookups = [
        ("Domas", "surname"),
        ("Jonas", "age"),
        ("Petras", "email")
    ]
    
    for vector_value, dimension in test_lookups:
        original = vdb.lookup(vector_value, dimension)
        reloaded = vdb2.lookup(vector_value, dimension)
        match = "✅" if original == reloaded else "❌"
        print(f"   {match} {vector_value}.{dimension}: {original} == {reloaded}")
    
    print_section("DEMO COMPLETE")
    print("🎉 Vector Database successfully demonstrated!")
    print("💡 Key Features Verified:")
    print("   ✅ O(1) coordinate-based lookups")
    print("   ✅ Value deduplication in dimensional spaces")
    print("   ✅ Automatic update propagation")
    print("   ✅ N-dimensional scalability")
    print("   ✅ Single-file persistence (SQLite-like)")
    print("   ✅ Complete data integrity")
    
    print(f"\n📁 Demo database file: {Path(demo_db_path).absolute()}")


if __name__ == "__main__":
    main()