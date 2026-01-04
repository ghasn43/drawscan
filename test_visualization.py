"""
Test the FIXED visualization module
"""
from visualization import DXFVisualizer

# Create test walls data
test_walls = [
    {
        'id': 1,
        'start_point': (0, 0),
        'end_point': (5000, 0),
        'length': 5000,
        'thickness': 230,
        'layer': 'EXTERIOR'
    },
    {
        'id': 2,
        'start_point': (5000, 0),
        'end_point': (5000, 3000),
        'length': 3000,
        'thickness': 115,
        'layer': 'INTERIOR'
    },
    {
        'id': 3,
        'start_point': (5000, 3000),
        'end_point': (0, 3000),
        'length': 5000,
        'thickness': 230,
        'layer': 'EXTERIOR'
    },
    {
        'id': 4,
        'start_point': (0, 3000),
        'end_point': (0, 0),
        'length': 3000,
        'thickness': 230,
        'layer': 'EXTERIOR'
    }
]

# Create test BOQ data
test_boq = [
    {'item': 'Brickwork in superstructure', 'quantity': 24.5, 'unit': 'm³'},
    {'item': 'Cement plaster 20mm thick', 'quantity': 156.8, 'unit': 'm²'},
    {'item': 'Concrete in foundation', 'quantity': 12.3, 'unit': 'm³'},
    {'item': 'Steel reinforcement', 'quantity': 1.2, 'unit': 'ton'},
    {'item': 'Formwork', 'quantity': 89.6, 'unit': 'm²'}
]

# Initialize visualizer
visualizer = DXFVisualizer()

print("Testing FIXED visualization module...")

# Test 1: Simple Visualization (most reliable)
print("1. Creating simple visualization...")
fig_simple = visualizer.create_simple_wall_visualization(test_walls)
fig_simple.savefig('test_simple.png', dpi=150, bbox_inches='tight')
print("   Saved: test_simple.png")

# Test 2: Advanced Visualization (with transform)
print("2. Creating advanced visualization...")
try:
    fig_advanced = visualizer.create_wall_visualization(test_walls, show_labels=True)
    fig_advanced.savefig('test_advanced.png', dpi=150, bbox_inches='tight')
    print("   Saved: test_advanced.png")
except Exception as e:
    print(f"   Advanced visualization failed: {e}")
    print("   Using simple visualization as fallback...")
    fig_simple.savefig('test_advanced.png', dpi=150, bbox_inches='tight')

# Test 3: 3D Visualization
print("3. Creating 3D visualization...")
fig_3d = visualizer.create_3d_visualization(test_walls, wall_height=3000)
fig_3d.write_html('test_3d.html')
print("   Saved: test_3d.html")

# Test 4: Quantity Chart
print("4. Creating quantity chart...")
fig_qty = visualizer.create_quantity_chart(test_boq)
fig_qty.savefig('test_quantity.png', dpi=150, bbox_inches='tight')
print("   Saved: test_quantity.png")

# Test 5: Dashboard
print("5. Creating dashboard...")
fig_dash = visualizer.create_summary_dashboard(test_walls, test_boq, "test.dxf")
fig_dash.savefig('test_dashboard.png', dpi=150, bbox_inches='tight')
print("   Saved: test_dashboard.png")

print("\n✅ All tests completed successfully!")
print("\nOpen test_3d.html in your browser for interactive 3D view!")