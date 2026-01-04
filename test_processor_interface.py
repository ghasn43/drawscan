"""
Test to understand the dxf_processor interface
"""
import dxf_processor

# Create processor instance
processor = dxf_processor.DXFProcessor()

# Check what each method does
print("DXFProcessor Method Details:")
print("=" * 50)

# Check load_file method
print("\n1. load_file method signature:")
try:
    import inspect
    sig = inspect.signature(processor.load_file)
    print(f"   Signature: {sig}")
except:
    print("   Could not get signature")

# Check extract_entities method
print("\n2. extract_entities method:")
try:
    sig = inspect.signature(processor.extract_entities)
    print(f"   Signature: {sig}")
except:
    print("   Could not get signature")

# Check calculate_quantities method
print("\n3. calculate_quantities method:")
try:
    sig = inspect.signature(processor.calculate_quantities)
    print(f"   Signature: {sig}")
except:
    print("   Could not get signature")

# Check what the methods return
print("\n\nTesting with a sample file...")
print("=" * 50)

try:
    # Try to load a test file if it exists
    import os
    test_files = ["test_building.dxf", "sample.dxf", "test.dxf"]
    
    test_file = None
    for file in test_files:
        if os.path.exists(file):
            test_file = file
            break
    
    if test_file:
        print(f"Testing with file: {test_file}")
        
        # Test load_file
        print("\nCalling load_file()...")
        result = processor.load_file(test_file)
        print(f"   Result type: {type(result)}")
        if result is not None:
            print(f"   Result: {result}")
        
        # Test extract_entities
        print("\nCalling extract_entities()...")
        entities = processor.extract_entities()
        print(f"   Entities type: {type(entities)}")
        if entities is not None:
            print(f"   Number of entities: {len(entities) if hasattr(entities, '__len__') else 'N/A'}")
            if entities and len(entities) > 0:
                print(f"   First entity: {entities[0] if hasattr(entities, '__getitem__') else 'N/A'}")
        
        # Test calculate_quantities
        print("\nCalling calculate_quantities()...")
        quantities = processor.calculate_quantities()
        print(f"   Quantities type: {type(quantities)}")
        if quantities is not None:
            print(f"   Quantities keys: {quantities.keys() if hasattr(quantities, 'keys') else 'N/A'}")
            
    else:
        print("No test DXF file found. Create one with 'python create_test_dxf.py'")
        
except Exception as e:
    print(f"Error during testing: {e}")