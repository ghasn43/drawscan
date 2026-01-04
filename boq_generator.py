#!/usr/bin/env python3
"""
BOQ Generator - Creates Bill of Quantities from extracted data
CORRECTED VERSION: Fixed all errors
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import json
from io import BytesIO
import math


class BOQGenerator:
    """Generates Bill of Quantities from extracted wall data"""
    
    # Standard rates (example - should be customized)
    STANDARD_RATES = {
        'concrete_walls': 8500,      # per m³
        'concrete_slabs': 8200,      # per m³
        'concrete_beams': 8500,      # per m³
        'concrete_columns': 8600,    # per m³
        'brick_walls': 4500,         # per m³
        'brickwork': 4500,           # per m³
        'doors': 12000,              # per piece
        'windows': 8000,             # per piece
        'plastering': 350,           # per m²
        'plaster': 350,              # per m²
        'painting': 180,             # per m²
        'flooring': 1200,            # per m²
    }
    
    # Standard dimensions
    STANDARDS = {
        'wall_thickness': 0.23,      # 230mm in meters
        'slab_thickness': 0.15,      # 150mm
        'beam_width': 0.23,          # 230mm
        'beam_depth': 0.45,          # 450mm
        'column_size': 0.23,         # 230mm square
        'door_width': 0.90,          # 900mm
        'door_height': 2.10,         # 2100mm
        'window_height': 1.20,       # 1200mm
    }
    
    def __init__(self, project_name: str = "DXF Project"):
        self.project_name = project_name
        self.boq_items = []
    
    def generate_boq(self, walls: List[Dict], wall_height: float = 3000.0) -> List[Dict]:
        """
        Generate BOQ from walls list
        
        Args:
            walls: List of wall dictionaries with 'length' in mm
            wall_height: Wall height in mm
            
        Returns:
            List of BOQ items
        """
        boq_data = []
        
        if not walls:
            return boq_data
        
        # Calculate totals from walls
        total_wall_length_m = sum(wall.get('length', 0) for wall in walls) / 1000  # Convert to meters
        wall_height_m = wall_height / 1000  # Convert to meters
        total_wall_area_m2 = total_wall_length_m * wall_height_m
        
        # Calculate average thickness
        avg_thickness_m = 0.23  # Default
        if walls:
            thicknesses = [wall.get('thickness', 230) for wall in walls]
            avg_thickness_m = sum(thicknesses) / len(thicknesses) / 1000
        
        # Calculate wall volume
        total_wall_volume_m3 = total_wall_area_m2 * avg_thickness_m
        
        # 1. Brickwork/Masonry
        if total_wall_volume_m3 > 0:
            boq_data.append({
                'item': 'Brickwork in superstructure (230mm thick)',
                'description': 'Brick masonry work in cement mortar 1:6',
                'quantity': round(total_wall_volume_m3, 3),
                'unit': 'm³',
                'rate': self.STANDARD_RATES['brickwork'],
                'total_cost': round(total_wall_volume_m3 * self.STANDARD_RATES['brickwork'], 2),
                'is_deduction': False,
                'is_summary': False
            })
        
        # 2. Plastering
        if total_wall_area_m2 > 0:
            # Internal plaster (both sides)
            boq_data.append({
                'item': 'Cement plaster 12mm thick (internal walls)',
                'description': '12mm thick cement plaster 1:4 on internal walls',
                'quantity': round(total_wall_area_m2 * 2, 3),  # Both sides
                'unit': 'm²',
                'rate': self.STANDARD_RATES['plaster'],
                'total_cost': round(total_wall_area_m2 * 2 * self.STANDARD_RATES['plaster'], 2),
                'is_deduction': False,
                'is_summary': False
            })
            
            # External plaster
            boq_data.append({
                'item': 'Cement plaster 20mm thick (external walls)',
                'description': '20mm thick weatherproof cement plaster 1:4 on external walls',
                'quantity': round(total_wall_area_m2, 3),  # One side
                'unit': 'm²',
                'rate': self.STANDARD_RATES['plaster'],
                'total_cost': round(total_wall_area_m2 * self.STANDARD_RATES['plaster'], 2),
                'is_deduction': False,
                'is_summary': False
            })
        
        # 3. Deductions for openings (estimated 15% of wall area)
        if total_wall_area_m2 > 0:
            opening_deduction = total_wall_area_m2 * 0.15
            if opening_deduction > 0:
                boq_data.append({
                    'item': 'Deduction for doors and windows',
                    'description': '15% deduction for openings in walls',
                    'quantity': round(opening_deduction, 3),
                    'unit': 'm²',
                    'rate': -self.STANDARD_RATES['plaster'],  # Negative rate for deduction
                    'total_cost': round(-opening_deduction * self.STANDARD_RATES['plaster'], 2),
                    'is_deduction': True,
                    'is_summary': False
                })
        
        # 4. Summary item
        boq_data.append({
            'item': 'TOTAL WALL LENGTH',
            'description': f'Total length of {len(walls)} extracted walls',
            'quantity': round(total_wall_length_m, 2),
            'unit': 'm',
            'rate': 0,
            'total_cost': 0,
            'is_deduction': False,
            'is_summary': True
        })
        
        return boq_data
    
    def calculate_costs(self, boq_data: List[Dict], 
                       brick_rate: float = None, 
                       plaster_rate: float = None) -> List[Dict]:
        """
        Calculate costs for BOQ items with customizable rates
        
        Args:
            boq_data: List of BOQ items
            brick_rate: Custom rate for brickwork per m³ (overrides default)
            plaster_rate: Custom rate for plaster per m² (overrides default)
            
        Returns:
            Updated BOQ data with costs
        """
        # Use custom rates if provided, otherwise use defaults
        brick_rate = brick_rate if brick_rate is not None else self.STANDARD_RATES['brickwork']
        plaster_rate = plaster_rate if plaster_rate is not None else self.STANDARD_RATES['plaster']
        
        for item in boq_data:
            item_name = item['item'].lower()
            
            # Skip summary items
            if item.get('is_summary', False):
                continue
            
            # Determine rate based on item type
            if 'brickwork' in item_name or 'masonry' in item_name:
                rate = brick_rate
            elif 'plaster' in item_name:
                rate = plaster_rate
            elif 'deduction' in item_name:
                rate = -plaster_rate  # Negative for deductions
            else:
                rate = item.get('rate', 0)
            
            # Calculate total cost
            quantity = item['quantity']
            total_cost = quantity * rate
            
            # Update item
            item['rate'] = rate
            item['total_cost'] = total_cost
        
        return boq_data
    
    def export_to_excel(self, boq_data: List[Dict]) -> BytesIO:
        """
        Export BOQ to Excel
        
        Args:
            boq_data: List of BOQ items
            
        Returns:
            BytesIO buffer with Excel file
        """
        # Convert to DataFrame
        df = pd.DataFrame(boq_data)
        
        # Create Excel writer
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Main BOQ sheet
            # Remove internal columns before saving
            export_df = df.drop(columns=['is_deduction', 'is_summary'], errors='ignore')
            export_df.to_excel(writer, sheet_name='BOQ', index=False)
            
            # Add summary sheet
            try:
                summary_data = self._create_summary(df)
                summary_df = pd.DataFrame([summary_data])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            except Exception as e:
                # Create a simple summary if detailed one fails
                print(f"Warning: Could not create detailed summary: {e}")
                simple_summary = {
                    'project_name': self.project_name,
                    'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_items': len(df),
                    'total_cost': df['total_cost'].sum() if 'total_cost' in df.columns else 0
                }
                summary_df = pd.DataFrame([simple_summary])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Formatting - set column widths for BOQ sheet
            worksheet = writer.sheets['BOQ']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if cell.value and len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        buffer.seek(0)
        return buffer
    
    def _create_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create summary statistics - FIXED VERSION"""
        if df.empty:
            return {}
        
        # Initialize masks with proper boolean Series
        summary_mask = pd.Series([False] * len(df), index=df.index)
        deduction_mask = pd.Series([False] * len(df), index=df.index)
        
        # Check if columns exist and create proper boolean masks
        if 'is_summary' in df.columns:
            summary_mask = df['is_summary'].fillna(False).astype(bool)
        
        if 'is_deduction' in df.columns:
            deduction_mask = df['is_deduction'].fillna(False).astype(bool)
        
        # Filter items using the boolean masks
        work_items = df[~summary_mask & ~deduction_mask].copy()
        deduction_items = df[deduction_mask].copy()
        
        # Calculate totals
        total_quantity = 0
        total_cost = 0
        total_deductions = 0
        
        if not work_items.empty and 'quantity' in work_items.columns:
            total_quantity = work_items['quantity'].sum()
        
        if not work_items.empty and 'total_cost' in work_items.columns:
            total_cost = work_items['total_cost'].sum()
        
        if not deduction_items.empty and 'total_cost' in deduction_items.columns:
            total_deductions = abs(deduction_items['total_cost'].sum())
        
        net_cost = total_cost - total_deductions
        
        # Count items
        item_count = len(work_items)
        
        # Get units used
        units = []
        if not work_items.empty and 'unit' in work_items.columns:
            units = work_items['unit'].dropna().unique().tolist()
        
        # Count specific item types
        brickwork_count = 0
        plaster_count = 0
        
        if not work_items.empty and 'item' in work_items.columns:
            brickwork_count = work_items['item'].str.contains('brickwork', case=False, na=False).sum()
            plaster_count = work_items['item'].str.contains('plaster', case=False, na=False).sum()
        
        return {
            'project_name': self.project_name,
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_work_items': int(item_count),
            'total_deduction_items': int(len(deduction_items)),
            'total_quantity': round(total_quantity, 3),
            'gross_cost': round(total_cost, 2),
            'total_deductions': round(total_deductions, 2),
            'net_cost': round(net_cost, 2),
            'units_used': ', '.join(units),
            'brickwork_items': int(brickwork_count),
            'plaster_items': int(plaster_count)
        }
    
    def export_to_json(self, boq_data: List[Dict]) -> str:
        """
        Export BOQ to JSON
        
        Args:
            boq_data: List of BOQ items
            
        Returns:
            JSON string
        """
        # Remove internal columns before exporting to JSON
        export_data = []
        for item in boq_data:
            export_item = item.copy()
            # Remove internal flags
            export_item.pop('is_deduction', None)
            export_item.pop('is_summary', None)
            export_data.append(export_item)
        
        return json.dumps(export_data, indent=2, default=str)
    
    def create_report(self, boq_data: List[Dict], output_path: Path = None) -> str:
        """
        Create a text report from BOQ data
        
        Args:
            boq_data: List of BOQ items
            output_path: Optional path to save report
            
        Returns:
            Report text
        """
        report_lines = []
        
        # Header
        report_lines.append(f"PROJECT: {self.project_name}")
        report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 70)
        report_lines.append("BILL OF QUANTITIES")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # BOQ Items
        item_no = 1
        total_cost = 0
        
        for item in boq_data:
            if item.get('is_summary', False):
                continue  # Skip summary rows
            
            description = item.get('description', '')
            quantity = item.get('quantity', 0)
            unit = item.get('unit', '')
            rate = item.get('rate', 0)
            cost = item.get('total_cost', 0)
            
            # Add to total (excluding deductions for net total)
            if not item.get('is_deduction', False):
                total_cost += cost
            
            report_lines.append(f"{item_no}. {item['item']}")
            if description:
                report_lines.append(f"   {description}")
            report_lines.append(f"   Quantity: {quantity} {unit}")
            report_lines.append(f"   Rate: ₹{rate:,.2f}")
            report_lines.append(f"   Amount: ₹{cost:,.2f}")
            report_lines.append("")
            
            item_no += 1
        
        # Summary
        report_lines.append("=" * 70)
        report_lines.append("SUMMARY")
        report_lines.append("=" * 70)
        report_lines.append(f"Total Items: {item_no - 1}")
        report_lines.append(f"Total Cost: ₹{total_cost:,.2f}")
        report_lines.append("")
        report_lines.append("END OF REPORT")
        
        report_text = "\n".join(report_lines)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text


# Utility functions
def generate_boq_from_walls(walls: List[Dict], wall_height: float = 3000.0) -> List[Dict]:
    """Quick utility function to generate BOQ"""
    generator = BOQGenerator()
    return generator.generate_boq(walls, wall_height)


def calculate_costs_for_boq(boq_data: List[Dict], brick_rate: float = 4500.0, 
                           plaster_rate: float = 350.0) -> List[Dict]:
    """Quick utility function to calculate costs"""
    generator = BOQGenerator()
    return generator.calculate_costs(boq_data, brick_rate, plaster_rate)


if __name__ == "__main__":
    # Test the generator
    print("Testing BOQ Generator...")
    print("=" * 60)
    
    # Create test walls
    test_walls = [
        {'id': 1, 'length': 5000, 'thickness': 230, 'layer': 'WALLS'},
        {'id': 2, 'length': 3000, 'thickness': 115, 'layer': 'PARTITION'},
        {'id': 3, 'length': 4000, 'thickness': 230, 'layer': 'WALLS'},
    ]
    
    generator = BOQGenerator("Test Project")
    
    # Test 1: Generate BOQ
    print("\n1. Testing generate_boq():")
    boq_data = generator.generate_boq(test_walls, wall_height=3000)
    print(f"   Generated {len(boq_data)} BOQ items")
    
    for i, item in enumerate(boq_data):
        print(f"   {i+1}. {item['item']}: {item['quantity']} {item['unit']}")
    
    # Test 2: Calculate costs with custom rates
    print("\n2. Testing calculate_costs() with custom rates:")
    boq_with_costs = generator.calculate_costs(boq_data, brick_rate=4800, plaster_rate=380)
    
    total_cost = 0
    for item in boq_with_costs:
        if 'total_cost' in item and not item.get('is_summary', False):
            cost = item['total_cost']
            if not item.get('is_deduction', False):
                total_cost += cost
            status = "(Deduction)" if item.get('is_deduction', False) else ""
            print(f"   {item['item']}: ₹{cost:,.2f} {status}")
    
    print(f"\n   Net Total Cost: ₹{total_cost:,.2f}")
    
    # Test 3: Export to Excel
    print("\n3. Testing export_to_excel():")
    try:
        excel_buffer = generator.export_to_excel(boq_with_costs)
        print(f"   ✓ Excel export successful")
        print(f"   Buffer size: {len(excel_buffer.getvalue())} bytes")
        
        # Test that we can read it back
        excel_buffer.seek(0)
        test_df = pd.read_excel(excel_buffer, sheet_name='BOQ')
        print(f"   ✓ Can read back {len(test_df)} rows")
    except Exception as e:
        print(f"   ✗ Excel export failed: {e}")
    
    # Test 4: Export to JSON
    print("\n4. Testing export_to_json():")
    try:
        json_str = generator.export_to_json(boq_with_costs)
        print(f"   ✓ JSON export successful")
        print(f"   JSON length: {len(json_str)} characters")
    except Exception as e:
        print(f"   ✗ JSON export failed: {e}")
    
    # Test 5: Create report
    print("\n5. Testing create_report():")
    try:
        report = generator.create_report(boq_with_costs)
        print(f"   ✓ Report created successfully")
        print(f"   Report length: {len(report.split('\\n'))} lines")
    except Exception as e:
        print(f"   ✗ Report creation failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")